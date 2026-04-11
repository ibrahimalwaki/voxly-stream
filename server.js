const http = require('http')
const { WebSocketServer } = require('ws')
const { createClient } = require('@deepgram/sdk')
const OpenAI = require('openai')

// Downsample 16-bit PCM from 24kHz to 8kHz (3:1) then convert to mulaw
function pcm24kTo8kMulaw(pcmBuffer) {
  const MULAW_BIAS = 33
  const MULAW_MAX = 0x1FFF
  const samples = pcmBuffer.length / 2
  const outSamples = Math.floor(samples / 3)
  const out = Buffer.alloc(outSamples)
  for (let i = 0; i < outSamples; i++) {
    // Simple decimation: take every 3rd sample
    let sample = pcmBuffer.readInt16LE(i * 6)
    const sign = (sample >> 8) & 0x80
    if (sign) sample = -sample
    if (sample > MULAW_MAX) sample = MULAW_MAX
    sample += MULAW_BIAS
    let exp = 7
    for (let expMask = 0x4000; (sample & expMask) === 0 && exp > 0; exp--, expMask >>= 1) {}
    const mantissa = (sample >> (exp + 3)) & 0x0F
    out[i] = ~(sign | (exp << 4) | mantissa) & 0xFF
  }
  return out
}

let _deepgram, _openai
function getDG() { return _deepgram || (_deepgram = createClient(process.env.DEEPGRAM_API_KEY)) }
function getOAI() { return _openai || (_openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })) }

const PORT = process.env.PORT || 8080

// ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are Aria, the AI receptionist for Bright Smile Dental. You speak like a warm, friendly, and professional human receptionist who makes every caller feel comfortable and taken care of.

## VOICE RULES — follow on every single turn
1. ONE question per response. Never ask two things at once.
2. Keep responses short (1–2 sentences). This is a natural phone call.
3. Avoid robotic or stiff phrasing. Speak naturally and conversationally.
4. Do NOT use filler phrases like "let me check", "one moment", or "I'm looking that up".
5. Do NOT open with "Certainly!", "Absolutely!", or overly formal language.
6. Always check conversation history — never ask for something already provided.
7. Use the caller's name naturally (max once after learning it).
8. Gently confirm what you heard before moving forward.
9. Tone should always feel calm, patient, and welcoming — never rushed or transactional.

If asked:
"I'm an AI receptionist — I can help with most things, and if you'd prefer a person I can connect you right away."

## PHONE NUMBER RULES — critical
- A valid phone number has 10 digits (or 11 with country code).
- If incomplete:
  → "I just want to make sure I got that right — could you share your full phone number?"
- Never guess or fill missing digits.
- If incorrect:
  → "Sorry about that — let's try again from the start."
- Always wait for confirmation before continuing.

## Clinic Info
- Name: Bright Smile Dental | Phone: (555) 123-4567
- Address: 123 Main Street, Suite 200, Springfield
- Hours: Mon–Wed 8am–6pm, Thu 8am–7pm, Fri 8am–5pm, Sat 9am–2pm, Sun closed
- Services: Cleanings, Fillings, Extractions, Root Canal, Crowns, Implants, Whitening, Veneers, Invisalign, Emergency Care, Pediatric Dentistry, Sedation

## BOOKING FLOW — one step at a time

STEP 1 — Full name
"Hi there! Can I get your full name?"
→ "Thanks, [name]."

STEP 2 — Phone number
"And what's the best number to reach you?"
→ "Just to confirm, that's [number] — did I get that right?"

STEP 3 — Service
"What can we help you with today — a cleaning, checkup, or something else?"
→ "Got it, a [service]."

STEP 4 — Time
"Do you have a day or time that works best for you?"
→ "Perfect, [day] at [time]."

STEP 5 — New patient
"Will this be your first visit with us?"
→ If yes: "Do you have dental insurance you'd like to use?"

STEP 6 — Confirmation
"Just to make sure everything looks good — [name], you're booked for a [service] on [day] at [time]. We'll call you at [number]. Does that all look right?"
→ If yes: "You're all set! We look forward to seeing you."

## DEMO PITCH (after booking — keep it light and natural)
"Before you go — just quickly, I'm Aria, an AI receptionist built by Voxly. What you just experienced is how clinics can handle calls 24/7 without missing patients. Would you be open to a quick 15-minute intro with our founder Ibrahim?"

### If YES:
"Great — what's the best number to reach you?"
→ confirm number
"Perfect — Ibrahim will reach out shortly. Have a great day!"

### If NO / hesitant:
"Totally understand — just to share, clinics using this are catching calls they used to miss, even after hours. It's usually worth a quick look, no pressure at all. Would that be okay?"
→ If still NO: "No problem at all — if you ever want to explore it, just let us know. Have a wonderful day!"

### If they say they already have a receptionist:
"That's great — this actually works alongside your team, not instead of them. It helps with missed calls and after-hours so your front desk isn't overwhelmed. Worth a quick 10-minute look?"

### If they ask how it works:
"It connects to your phone line and booking system, and Ibrahim handles everything for you. Want me to grab your details so he can reach out?"

## Transfer to Human
Only if caller explicitly says "talk to a person", "transfer me", "speak to someone", "real person", or "human".
Say: "Of course — press 9 on your keypad and we'll connect you right away." then output: [TRANSFER_TO_HUMAN]`

// ── HTTP SERVER (for Twilio webhook + health check) ───────────────────────────
const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200)
    res.end('ok')
    return
  }

  if (req.method === 'POST' && req.url === '/voice') {
    // Twilio calls this when a call comes in — respond with TwiML to start stream
    const host = req.headers.host
    const twiml = `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://${host}/stream" />
  </Connect>
</Response>`
    res.writeHead(200, { 'Content-Type': 'text/xml' })
    res.end(twiml)
    return
  }

  res.writeHead(404)
  res.end()
})

// ── WEBSOCKET SERVER ──────────────────────────────────────────────────────────
const wss = new WebSocketServer({ server, path: '/stream' })

wss.on('connection', (twilioWs) => {
  console.log('Twilio connected')

  let streamSid = null
  let callSid = null
  let conversationHistory = []
  let dgLive = null
  let dgReady = false
  let pendingAudioFrames = []
  let isProcessing = false
  let isPlaying = false
  let speechBuffer = ''
  let silenceTimer = null
  let clientConfig = null // loaded per-call based on Twilio number

  // ── DEEPGRAM LIVE STT ──────────────────────────────────────────────────────
  async function startDeepgram() {
    const connection = getDG().listen.live({
      model: 'nova-3',
      language: 'en-US',
      smart_format: true,
      interim_results: false,
      vad_events: true,
      encoding: 'mulaw',
      sample_rate: 8000,
    })

    connection.on('open', () => {
      console.log('Deepgram connected')
      dgReady = true

      if (pendingAudioFrames.length > 0) {
        for (const frame of pendingAudioFrames) {
          connection.send(frame)
        }
        console.log(`Flushed ${pendingAudioFrames.length} queued audio frames to Deepgram`)
        pendingAudioFrames = []
      }
    })

    connection.on('Results', (data) => {
      const transcript = data.channel?.alternatives?.[0]?.transcript
      if (!transcript || !data.is_final) return

      speechBuffer += ' ' + transcript
      speechBuffer = speechBuffer.trim()

      // Barge-in: if caller speaks while Aria is talking, stop Aria immediately
      if (isPlaying && streamSid && twilioWs.readyState === twilioWs.OPEN) {
        twilioWs.send(JSON.stringify({ event: 'clear', streamSid }))
        isPlaying = false
        isProcessing = false
      }

      // Use longer delay when caller is spelling (single letter fragments)
      const isSpelling = transcript.trim().length <= 1
      const delay = isSpelling ? 1500 : 700

      clearTimeout(silenceTimer)
      silenceTimer = setTimeout(() => {
        if (speechBuffer && !isProcessing) {
          const text = speechBuffer
          speechBuffer = ''
          handleUserSpeech(text)
        }
      }, delay)
    })

    connection.on('UtteranceEnd', () => {
      // Fallback: only fire if buffer looks complete (not mid-spelling)
      if (speechBuffer && !isProcessing && speechBuffer.length >= 5) {
        clearTimeout(silenceTimer)
        const text = speechBuffer
        speechBuffer = ''
        handleUserSpeech(text)
      }
    })

    connection.on('error', (err) => {
      console.error('Deepgram error:', {
        message: err?.message,
        type: err?.type,
        code: err?.code,
        reason: err?.reason,
      })
    })
    connection.on('close', () => {
      dgReady = false
      pendingAudioFrames = []
      console.log('Deepgram closed')
    })

    return connection
  }

  // ── EXTRACT BOOKING DATA AND POST TO DENTAL SAAS ──────────────────────────
  async function saveBooking(history) {
    const appUrl = process.env.VOXLY_APP_URL
    if (!appUrl) return

    const transcript = history.map(m => `${m.role === 'assistant' ? 'Aria' : 'Caller'}: ${m.content}`).join('\n')

    const extractResult = await getOAI().chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 200,
      temperature: 0,
      messages: [
        { role: 'system', content: 'Extract booking details from a call transcript. Return ONLY valid JSON with keys: patient_name, patient_phone, service, preferred_time, notes. Use null for missing values.' },
        { role: 'user', content: transcript },
      ],
    })

    let booking
    try {
      const raw = extractResult.choices[0]?.message?.content?.replace(/```json|```/g, '').trim()
      booking = JSON.parse(raw)
    } catch {
      console.error('Failed to parse booking JSON from Gemini')
      return
    }

    const res = await fetch(`${appUrl}/api/appointments/demo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(booking),
    })

    if (res.ok) {
      console.log('Booking saved to Supabase:', booking.patient_name)
    } else {
      console.error('Booking API error:', res.status, await res.text())
    }
  }

  // ── FETCH TTS AUDIO → mulaw buffer ────────────────────────────────────────
  async function fetchTTSAudio(text) {
    const response = await getOAI().audio.speech.create({
      model: 'tts-1',
      voice: clientConfig?.voice || 'nova',
      input: text,
      response_format: 'pcm',
      speed: 1.0,
    })
    const pcmBuffer = Buffer.from(await response.arrayBuffer())
    return pcm24kTo8kMulaw(pcmBuffer)
  }

  // ── PLAY AUDIO BUFFERS IN ORDER ────────────────────────────────────────────
  // Accepts an array of Promise<Buffer> — fires all in parallel, plays in order.
  async function playAudioQueue(audioPromises) {
    if (!streamSid || twilioWs.readyState !== twilioWs.OPEN) return
    isPlaying = true
    const chunkSize = 160
    for (const audioPromise of audioPromises) {
      if (!isPlaying) break
      const audioBuffer = await audioPromise
      if (!audioBuffer) continue
      for (let i = 0; i < audioBuffer.length; i += chunkSize) {
        if (!isPlaying) break
        twilioWs.send(JSON.stringify({
          event: 'media',
          streamSid,
          media: { payload: audioBuffer.subarray(i, i + chunkSize).toString('base64') },
        }))
      }
    }
    if (streamSid && twilioWs.readyState === twilioWs.OPEN) {
      twilioWs.send(JSON.stringify({ event: 'mark', streamSid, mark: { name: 'end_of_response' } }))
    }
  }

  // ── STREAM GPT → SENTENCE-LEVEL TTS PIPELINE ──────────────────────────────
  // Streams GPT tokens, fires TTS the moment each sentence is complete,
  // plays all sentences in order. First audio arrives in ~sentence_1_time
  // instead of full_response_time + full_tts_time.
  async function streamGPTAndSpeak(messages) {
    const stream = await getOAI().chat.completions.create({
      model: 'gpt-4o-mini',
      max_tokens: 120,
      temperature: 0.6,
      messages,
      stream: true,
    })

    let tokenBuffer = ''
    let fullText = ''
    const audioQueue = [] // Promise<Buffer>[] — TTS fires immediately, played in order

    function flushSentence(sentence) {
      sentence = sentence.trim()
      if (sentence.length < 2) return
      // Fire TTS immediately (non-blocking) — result queued for ordered playback
      audioQueue.push(fetchTTSAudio(sentence))
    }

    for await (const chunk of stream) {
      const token = chunk.choices[0]?.delta?.content || ''
      if (!token) continue
      tokenBuffer += token
      fullText += token

      // Split on sentence-ending punctuation followed by whitespace
      // Also split on em-dash pause if buffer is getting long (natural breath point)
      const boundary = tokenBuffer.match(/^(.+?[.!?])\s+/) ||
        (tokenBuffer.length > 80 ? tokenBuffer.match(/^(.+?[,—])\s+/) : null)

      if (boundary) {
        flushSentence(boundary[1])
        tokenBuffer = tokenBuffer.slice(boundary[0].length)
      }
    }

    // Flush anything left in the buffer
    if (tokenBuffer.trim()) flushSentence(tokenBuffer)

    // Play all queued audio in order (TTS for later sentences is already running)
    await playAudioQueue(audioQueue)

    return fullText
  }

  // ── HANDLE USER SPEECH ─────────────────────────────────────────────────────
  async function handleUserSpeech(text) {
    if (!text.trim() || isProcessing) return
    isProcessing = true
    console.log('User:', text)

    conversationHistory.push({ role: 'user', content: text })

    try {
      const activePrompt = clientConfig?.system_prompt || SYSTEM_PROMPT
      const now = new Date().toLocaleString('en-US', {
        timeZone: 'America/Toronto',
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
        hour: 'numeric', minute: '2-digit',
      })
      const baseMessages = [
        { role: 'system', content: `${activePrompt}\n\nCurrent date and time: ${now} (Eastern Time)` },
        ...conversationHistory,
      ]

      // Only run the tool-call loop if the message might contain a time/date.
      // Everything else goes straight to the streaming pipeline.
      const mightNeedTools = /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|morning|afternoon|evening|next week|\d+\s*(am|pm|:))/i.test(text)

      let aiText = ''

      if (mightNeedTools) {
        // ── Tool-call path (non-streaming) ──────────────────────────────────
        const tools = [
          {
            type: 'function',
            function: {
              name: 'check_slot_availability',
              description: 'Check if a requested appointment time is available and within clinic hours. Call this BEFORE confirming any appointment time with the caller.',
              parameters: {
                type: 'object',
                properties: {
                  datetime: { type: 'string', description: 'The requested date and time as a natural language string (e.g. "Thursday at 2pm", "next Monday morning")' },
                },
                required: ['datetime'],
              },
            },
          },
        ]

        let messages = baseMessages
        for (let i = 0; i < 3; i++) {
          const completion = await getOAI().chat.completions.create({
            model: 'gpt-4o-mini', max_tokens: 120, temperature: 0.6, messages, tools, tool_choice: 'auto',
          })
          const choice = completion.choices[0]
          const msg = choice.message

          if (choice.finish_reason === 'tool_calls' && msg.tool_calls?.length > 0) {
            messages.push(msg)
            for (const toolCall of msg.tool_calls) {
              let toolResult
              try {
                const args = JSON.parse(toolCall.function.arguments)
                const appUrl = process.env.VOXLY_APP_URL
                if (appUrl) {
                  const res = await fetch(`${appUrl}/api/availability?datetime=${encodeURIComponent(args.datetime)}`)
                  toolResult = await res.json()
                } else {
                  toolResult = { available: true }
                }
              } catch {
                toolResult = { available: true }
              }
              console.log('Availability check:', toolResult)
              messages.push({ role: 'tool', tool_call_id: toolCall.id, content: JSON.stringify(toolResult) })
            }
          } else {
            aiText = msg.content || "I'm sorry, could you repeat that?"
            break
          }
        }
        if (!aiText) aiText = "I'm sorry, could you repeat that?"

        // Stream TTS for the tool-call result
        const clean = aiText.replace('[TRANSFER_TO_HUMAN]', '').trim()
        await playAudioQueue([fetchTTSAudio(clean)])
      } else {
        // ── Streaming pipeline path (most turns) ───────────────────────────
        aiText = await streamGPTAndSpeak(baseMessages)
      }

      const transferRequested = aiText.includes('[TRANSFER_TO_HUMAN]')
      const cleanText = aiText.replace('[TRANSFER_TO_HUMAN]', '').trim()

      console.log('Aria:', cleanText)
      conversationHistory.push({ role: 'assistant', content: cleanText })

      if (cleanText.toLowerCase().includes("you're all set")) {
        saveBooking(conversationHistory).catch(err => console.error('Booking save failed:', err))
      }

      if (transferRequested) {
        setTimeout(() => {
          if (twilioWs.readyState === twilioWs.OPEN) {
            twilioWs.send(JSON.stringify({ event: 'stop', streamSid }))
          }
        }, 1000)
      }

    } catch (err) {
      console.error('Pipeline error:', err)
    } finally {
      isProcessing = false
    }
  }

  // ── GREETING ───────────────────────────────────────────────────────────────
  async function sendGreeting() {
    const clinicName = clientConfig?.name || 'Bright Smile Dental'
    const aiName = clientConfig?.ai_name || 'Aria'
    const greeting = `Hello, thank you for calling ${clinicName} — this is ${aiName}, how can I help you today?`
    conversationHistory.push({ role: 'assistant', content: greeting })

    try {
      const ttsResponse = await getOAI().audio.speech.create({
        model: 'tts-1',
        voice: clientConfig?.voice || 'nova',
        input: greeting,
        response_format: 'pcm',
        speed: 1.0,
      })

      const pcmBuffer = Buffer.from(await ttsResponse.arrayBuffer())
      const audioBuffer = pcm24kTo8kMulaw(pcmBuffer)

      if (streamSid && twilioWs.readyState === twilioWs.OPEN) {
        const chunkSize = 160
        for (let i = 0; i < audioBuffer.length; i += chunkSize) {
          const chunk = audioBuffer.subarray(i, i + chunkSize)
          twilioWs.send(JSON.stringify({
            event: 'media',
            streamSid,
            media: { payload: chunk.toString('base64') },
          }))
        }
        twilioWs.send(JSON.stringify({
          event: 'mark',
          streamSid,
          mark: { name: 'greeting_end' },
        }))
      }
    } catch (err) {
      console.error('Greeting error:', err)
    }
  }

  // ── TWILIO MESSAGE HANDLER ─────────────────────────────────────────────────
  twilioWs.on('message', async (data) => {
    let msg
    try { msg = JSON.parse(data) } catch { return }

    switch (msg.event) {
      case 'start':
        streamSid = msg.start.streamSid
        callSid = msg.start.callSid
        console.log('Stream started:', streamSid)

        // Load client config based on which Twilio number was called
        try {
          const toNumber = msg.start.customParameters?.to || msg.start.to
          if (toNumber && process.env.VOXLY_APP_URL) {
            const res = await fetch(`${process.env.VOXLY_APP_URL}/api/clients?phone=${encodeURIComponent(toNumber)}`)
            if (res.ok) {
              clientConfig = await res.json()
              console.log('Loaded client config:', clientConfig.name)
            }
          }
        } catch (err) {
          console.error('Failed to load client config:', err)
        }

        dgLive = await startDeepgram()
        setTimeout(() => sendGreeting(), 500)
        break

      case 'media':
        if (dgLive) {
          const audioData = Buffer.from(msg.media.payload, 'base64')
          if (dgReady) {
            dgLive.send(audioData)
          } else {
            pendingAudioFrames.push(audioData)
          }
        }
        break

      case 'mark':
        if (msg.mark?.name === 'end_of_response') isPlaying = false
        break

      case 'stop':
        console.log('Stream stopped')
        if (dgLive) dgLive.finish()
        break
    }
  })

  twilioWs.on('close', () => {
    console.log('Twilio disconnected')
    if (dgLive) dgLive.finish()
    clearTimeout(silenceTimer)
  })

  twilioWs.on('error', (err) => console.error('Twilio WS error:', err))
})

server.listen(PORT, () => {
  console.log(`Voxly stream server running on port ${PORT}`)
})
