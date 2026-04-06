const http = require('http')
const { WebSocketServer } = require('ws')
const { createClient } = require('@deepgram/sdk')
const Groq = require('groq-sdk')
const OpenAI = require('openai')

// Downsample 16-bit PCM from 24kHz to 8kHz (3:1) then convert to mulaw
function pcm24kTo8kMulaw(pcmBuffer) {
  const MULAW_BIAS = 33
  const MULAW_MAX = 0x1FFF
  const samples = pcmBuffer.length / 2
  const outSamples = Math.floor(samples / 3)
  const out = Buffer.alloc(outSamples)
  for (let i = 0; i < outSamples; i++) {
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

let _deepgram, _groq, _openai
function getDG() { return _deepgram || (_deepgram = createClient(process.env.DEEPGRAM_API_KEY)) }
function getGroq() { return _groq || (_groq = new Groq({ apiKey: process.env.GROQ_API_KEY })) }
function getOAI() { return _openai || (_openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })) }

const PORT = process.env.PORT || 8080

// ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are Aria, the AI receptionist for Bright Smile Dental. You handle patient calls exactly like a warm, experienced human receptionist.

## VOICE RULES — follow on every single turn
1. ONE question per response. Never ask two things at once.
2. 1–2 sentences max. This is a phone call not an email.
3. Never say "let me check", "one moment", "I'm looking that up", or any filler.
4. Never open with "Certainly!", "Absolutely!", "Great!" — respond naturally.
5. Always check conversation history before asking — never re-ask something already answered.
6. Use the caller's name at most once after learning it.
7. Confirm what you just heard, then ask only the next missing piece.
8. Never say you are an AI unless directly asked. If asked say: "I'm an AI receptionist — I can help with most things, and if you'd prefer a person I can connect you right away."

## Clinic Info
- Name: Bright Smile Dental | Phone: (555) 123-4567
- Address: 123 Main Street, Suite 200, Springfield
- Hours: Mon–Wed 8am–6pm, Thu 8am–7pm, Fri 8am–5pm, Sat 9am–2pm, Sun closed
- Services: Cleanings, Fillings, Extractions, Root Canal, Crowns, Implants, Whitening, Veneers, Invisalign, Emergency Care, Pediatric Dentistry, Sedation

## BOOKING FLOW — strictly in this order, one step at a time
You are collecting information to book an appointment. Do NOT skip steps. Do NOT move to the next step until you have confirmed the current one.

STEP 1 — Full name
  Ask: "Can I get your full name?"
  When given: confirm it. "Got it, [name]." → go to step 2.

STEP 2 — Callback number
  Ask: "And the best number to reach you?"
  When given: repeat it back. "So that's [number] — is that right?"
  Wait for confirmation before moving on.

STEP 3 — Email address
  Ask: "What's the best email for your confirmation?"
  When given: spell it back letter by letter. Example — they say "jackson at gmail dot com" → you say "Let me just confirm: j-a-c-k-s-o-n at gmail dot com — is that right?"
  Wait for them to say yes before moving on.

STEP 4 — Service needed
  Ask: "What brings you in — is this for a cleaning, checkup, or something else?"
  When given: confirm it. "Got it, a [service]." → go to step 5.

STEP 5 — Preferred day and time
  Ask: "Do you have a preferred day or time in mind?"
  When given: confirm it. "Perfect, [day] at [time]." → go to step 6.

STEP 6 — New patient insurance check
  Ask: "Is this your first visit with us?"
  If yes: "Do you have dental insurance you'd like to use?"
  If no: skip this step.

STEP 7 — Final read-back and confirmation
  Read everything back in one sentence: "Just to confirm — [name], a [service] on [day] at [time]. We'll call you at [number] and send confirmation to [email]. Does that all sound right?"
  If they say something is wrong: "My apologies — what did I get wrong?" then correct and re-confirm.
  WAIT for the caller to say yes/correct/sounds good before saying "You're all set!"
  Only say "You're all set!" after explicit confirmation.

STEP 8 — Demo pitch (always do this after "You're all set!")
  Say: "Before I let you go — I want to share something with you. I'm Aria, an AI receptionist built by Voxly. What you just experienced is exactly what your clinic's patients would hear 24 hours a day, 7 days a week. Most dental clinics miss 5 to 10 calls a day — I answer every single one. Would you be open to a quick 15-minute call with our founder Ibrahim?"

  If yes:
    Ask: "What's the best number to reach you?"
    Wait for number → repeat it back: "So that's [number] — is that right?"
    Wait for confirmation → then ask: "And the best email for the calendar invite?"
    Wait for email → spell it back letter by letter: "Let me confirm — [spell each letter] — is that right?"
    Wait for confirmation → say: "Perfect — Ibrahim will be in touch shortly. Have a great day!"

  If no / not interested:
    Say: "Totally understand — I'll be honest with you though, clinics that looked into this are seeing real results. Patients getting booked at 2am, zero missed calls, front desk less stressed. It's genuinely worth even 10 minutes of your time. No commitment at all — just a quick look. Would that be okay?"
    If still no: "No problem at all — if you ever change your mind just call back anytime. Have a wonderful day!"

  If they say they already have a receptionist:
    Say: "That's great — this actually works alongside your team, not instead of them. It handles the overflow and after-hours calls they simply can't get to. Your receptionist gets to focus on the patients right in front of them. Worth even a quick 10 minutes with Ibrahim?"

  If they ask how it works:
    Say: "It plugs directly into your phone line and booking system — Ibrahim handles the entire setup for you. Want me to grab your number and email so he can reach out?"

## Transfer to Human
Only if caller explicitly says "talk to a person", "transfer me", "speak to someone", "real person", "human".
Say: "Of course — press 9 on your keypad to be connected right now." then output: [TRANSFER_TO_HUMAN]`

// ── HTTP SERVER ───────────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200)
    res.end('ok')
    return
  }

  if (req.method === 'POST' && req.url === '/voice') {
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

  // ── STREAM TTS → TWILIO (sends audio as it arrives, no waiting) ────────────
  async function streamTtsToTwilio(text) {
    if (!text.trim() || !streamSid || twilioWs.readyState !== twilioWs.OPEN) return

    const response = await getOAI().audio.speech.create({
      model: 'gpt-4o-mini-tts',
      voice: 'marin',
      input: text,
      response_format: 'pcm',
      speed: 1.0,
    })

    let leftover = Buffer.alloc(0)
    for await (const rawChunk of response.body) {
      const buf = Buffer.concat([leftover, Buffer.from(rawChunk)])
      // Need multiples of 6 bytes (3 input samples × 2 bytes per sample → 1 output sample)
      const processable = Math.floor(buf.length / 6) * 6
      if (processable > 0) {
        const mulaw = pcm24kTo8kMulaw(buf.subarray(0, processable))
        sendMulawToTwilio(mulaw)
        leftover = buf.subarray(processable)
      } else {
        leftover = buf
      }
    }
    // Flush remaining bytes (pad to multiple of 6)
    if (leftover.length > 0) {
      const padded = Buffer.alloc(Math.ceil(leftover.length / 6) * 6)
      leftover.copy(padded)
      sendMulawToTwilio(pcm24kTo8kMulaw(padded))
    }
  }

  function sendMulawToTwilio(audioBuffer) {
    if (!streamSid || twilioWs.readyState !== twilioWs.OPEN) return
    const chunkSize = 160 // 20ms at 8kHz
    for (let i = 0; i < audioBuffer.length; i += chunkSize) {
      twilioWs.send(JSON.stringify({
        event: 'media',
        streamSid,
        media: { payload: audioBuffer.subarray(i, i + chunkSize).toString('base64') },
      }))
    }
  }

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
        for (const frame of pendingAudioFrames) connection.send(frame)
        console.log(`Flushed ${pendingAudioFrames.length} queued frames`)
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
      }, 500)
    })

    connection.on('UtteranceEnd', () => {
      if (speechBuffer && !isProcessing) {
        clearTimeout(silenceTimer)
        const text = speechBuffer
        speechBuffer = ''
        handleUserSpeech(text)
      }
    })

    connection.on('error', (err) => {
      console.error('Deepgram error:', { message: err?.message, code: err?.code })
    })
    connection.on('close', () => {
      dgReady = false
      pendingAudioFrames = []
      console.log('Deepgram closed')
    })

    return connection
  }

  // ── SAVE BOOKING TO DENTAL SAAS ────────────────────────────────────────────
  async function saveBooking(history) {
    const appUrl = process.env.VOXLY_APP_URL
    if (!appUrl) return

    const transcript = history.map(m => `${m.role === 'assistant' ? 'Aria' : 'Caller'}: ${m.content}`).join('\n')

    const extractResult = await getGroq().chat.completions.create({
      model: 'llama-3.3-70b-versatile',
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
      console.error('Failed to parse booking JSON')
      return
    }

    const res = await fetch(`${appUrl}/api/appointments/demo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(booking),
    })

    if (res.ok) console.log('Booking saved:', booking.patient_name)
    else console.error('Booking API error:', res.status, await res.text())
  }

  // ── HANDLE USER SPEECH: stream Groq → sentence TTS → Twilio ───────────────
  async function handleUserSpeech(text) {
    if (!text.trim() || isProcessing) return
    isProcessing = true
    console.log('User:', text)

    conversationHistory.push({ role: 'user', content: text })

    try {
      // Stream Groq tokens
      const groqStream = await getGroq().chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        max_tokens: 150,
        temperature: 0.6,
        stream: true,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          ...conversationHistory,
        ],
      })

      let tokenBuffer = ''
      let fullAiText = ''
      // TTS sentences run sequentially via promise chain
      let ttsChain = Promise.resolve()

      function flushSentence(sentence) {
        sentence = sentence.replace('[TRANSFER_TO_HUMAN]', '').trim()
        if (!sentence) return
        ttsChain = ttsChain.then(() => streamTtsToTwilio(sentence))
      }

      for await (const chunk of groqStream) {
        const token = chunk.choices[0]?.delta?.content || ''
        if (!token) continue
        tokenBuffer += token
        fullAiText += token

        // Flush complete sentences as they arrive
        const sentenceRegex = /[^.!?]+[.!?]+\s*/g
        let match
        let lastIndex = 0
        while ((match = sentenceRegex.exec(tokenBuffer)) !== null) {
          flushSentence(match[0])
          lastIndex = sentenceRegex.lastIndex
        }
        tokenBuffer = tokenBuffer.slice(lastIndex)
      }

      // Flush any remaining text (no sentence-ending punctuation)
      if (tokenBuffer.trim()) flushSentence(tokenBuffer)

      // Wait for all TTS to finish
      await ttsChain

      // Send end mark
      if (streamSid && twilioWs.readyState === twilioWs.OPEN) {
        twilioWs.send(JSON.stringify({ event: 'mark', streamSid, mark: { name: 'end_of_response' } }))
      }

      fullAiText = fullAiText.trim()
      const transferRequested = fullAiText.includes('[TRANSFER_TO_HUMAN]')
      fullAiText = fullAiText.replace('[TRANSFER_TO_HUMAN]', '').trim()

      console.log('Aria:', fullAiText)
      conversationHistory.push({ role: 'assistant', content: fullAiText })

      if (fullAiText.toLowerCase().includes("you're all set")) {
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
    const greeting = "Hello, thank you for calling Bright Smile Dental — this is Aria, how can I help you today?"
    conversationHistory.push({ role: 'assistant', content: greeting })
    try {
      await streamTtsToTwilio(greeting)
      if (streamSid && twilioWs.readyState === twilioWs.OPEN) {
        twilioWs.send(JSON.stringify({ event: 'mark', streamSid, mark: { name: 'greeting_end' } }))
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
        dgLive = await startDeepgram()
        setTimeout(() => sendGreeting(), 500)
        break

      case 'media':
        if (dgLive) {
          const audioData = Buffer.from(msg.media.payload, 'base64')
          if (dgReady) dgLive.send(audioData)
          else pendingAudioFrames.push(audioData)
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
