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

STEP 3 — Email
"What's the best email to send your confirmation?"
→ "Let me confirm: [spell it out] — is that correct?"

STEP 4 — Service
"What can we help you with today — a cleaning, checkup, or something else?"
→ "Got it, a [service]."

STEP 5 — Time
"Do you have a day or time that works best for you?"
→ "Perfect, [day] at [time]."

STEP 6 — New patient
"Will this be your first visit with us?"
→ If yes: "Do you have dental insurance you'd like to use?"

STEP 7 — Confirmation
"Just to make sure everything looks good — [name], you're booked for a [service] on [day] at [time]. We'll call you at [number] and send details to [email]. Does that all look right?"
→ If yes: "You're all set! We look forward to seeing you."

## DEMO PITCH (after booking — keep it light and natural)
"Before you go — just quickly, I'm Aria, an AI receptionist built by Voxly. What you just experienced is how clinics can handle calls 24/7 without missing patients. Would you be open to a quick 15-minute intro with our founder Ibrahim?"

### If YES:
"Great — what's the best number to reach you?"
→ confirm number
"And the best email for the invite?"
→ confirm email
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

      clearTimeout(silenceTimer)
      silenceTimer = setTimeout(() => {
        if (speechBuffer && !isProcessing) {
          const text = speechBuffer
          speechBuffer = ''
          handleUserSpeech(text)
        }
      }, 700)
    })

    connection.on('UtteranceEnd', () => {
      // Fallback in case Deepgram emits utterance end after the final result timer
      if (speechBuffer && !isProcessing) {
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
        { role: 'system', content: 'Extract booking details from a call transcript. Return ONLY valid JSON with keys: patient_name, patient_phone, patient_email, service, preferred_time, notes. Use null for missing values.' },
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

  // ── HANDLE USER SPEECH → GEMINI → TTS → TWILIO ──────────────────────────────
  async function handleUserSpeech(text) {
    if (!text.trim() || isProcessing) return
    isProcessing = true
    console.log('User:', text)

    conversationHistory.push({ role: 'user', content: text })

    try {
      // 1. Get AI response from OpenAI
      const activePrompt = clientConfig?.system_prompt || SYSTEM_PROMPT
      const completion = await getOAI().chat.completions.create({
        model: 'gpt-4o-mini',
        max_tokens: 120,
        temperature: 0.6,
        messages: [
          { role: 'system', content: activePrompt },
          ...conversationHistory,
        ],
      })

      let aiText = completion.choices[0]?.message?.content || "I'm sorry, could you repeat that?"
      const transferRequested = aiText.includes('[TRANSFER_TO_HUMAN]')
      aiText = aiText.replace('[TRANSFER_TO_HUMAN]', '').trim()

      console.log('Aria:', aiText)
      conversationHistory.push({ role: 'assistant', content: aiText })

      // If booking just confirmed, extract and save it
      if (aiText.toLowerCase().includes("you're all set")) {
        saveBooking(conversationHistory).catch(err => console.error('Booking save failed:', err))
      }

      // 2. Generate TTS audio from OpenAI
      const ttsResponse = await getOAI().audio.speech.create({
        model: 'tts-1',
        voice: clientConfig?.voice || 'nova',
        input: aiText,
        response_format: 'pcm',
        speed: 1.0,
      })

      const pcmBuffer = Buffer.from(await ttsResponse.arrayBuffer())
      const audioBuffer = pcm24kTo8kMulaw(pcmBuffer)

      // 3. Send audio back to Twilio in chunks
      if (streamSid && twilioWs.readyState === twilioWs.OPEN) {
        const chunkSize = 160 // 20ms of audio at 8kHz mulaw
        for (let i = 0; i < audioBuffer.length; i += chunkSize) {
          const chunk = audioBuffer.subarray(i, i + chunkSize)
          twilioWs.send(JSON.stringify({
            event: 'media',
            streamSid,
            media: {
              payload: chunk.toString('base64'),
            },
          }))
        }

        // Mark end of audio
        twilioWs.send(JSON.stringify({
          event: 'mark',
          streamSid,
          mark: { name: 'end_of_response' },
        }))
      }

      // Handle transfer
      if (transferRequested) {
        setTimeout(() => {
          if (twilioWs.readyState === twilioWs.OPEN) {
            twilioWs.send(JSON.stringify({
              event: 'stop',
              streamSid,
            }))
          }
        }, audioBuffer.length * 1000 / 8000 + 500)
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
