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
  WAIT for the caller to say yes/correct/sounds good before saying "You're all set!"
  Only say "You're all set!" after explicit confirmation.

## STEP 8 — Demo pitch (always do this after "You're all set!")
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
  let isProcessing = false
  let speechBuffer = ''
  let silenceTimer = null

  // ── DEEPGRAM LIVE STT ──────────────────────────────────────────────────────
  async function startDeepgram() {
    const connection = getDG().listen.live({
      model: 'nova-3',
      language: 'en-US',
      smart_format: true,
      interim_results: true,
      utterance_end_ms: 1000,
      vad_events: true,
      encoding: 'mulaw',
      sample_rate: 8000,
    })

    connection.on('open', () => {
      console.log('Deepgram connected')
    })

    connection.on('Results', (data) => {
      const transcript = data.channel?.alternatives?.[0]?.transcript
      if (!transcript || !data.is_final) return

      speechBuffer += ' ' + transcript
      speechBuffer = speechBuffer.trim()

      // Only trigger early if the sentence sounds complete (ends with punctuation
      // or is a short clear answer like "yes", "no", a number, or a name)
      const looksComplete = /[.!?]$/.test(speechBuffer) ||
        /^(yes|no|yeah|nope|sure|okay|ok|correct|right|that'?s? right|sounds good)$/i.test(speechBuffer.trim()) ||
        speechBuffer.trim().split(' ').length <= 4

      clearTimeout(silenceTimer)
      silenceTimer = setTimeout(() => {
        if (speechBuffer && !isProcessing) {
          const text = speechBuffer
          speechBuffer = ''
          handleUserSpeech(text)
        }
      }, looksComplete ? 400 : 900) // short answers: 400ms, longer speech: 900ms
    })

    connection.on('UtteranceEnd', () => {
      // Always fires after 1s silence — catches anything the timer missed
      if (speechBuffer && !isProcessing) {
        clearTimeout(silenceTimer)
        const text = speechBuffer
        speechBuffer = ''
        handleUserSpeech(text)
      }
    })

    connection.on('error', (err) => console.error('Deepgram error:', err))
    connection.on('close', () => console.log('Deepgram closed'))

    return connection
  }

  // ── HANDLE USER SPEECH → GROQ → TTS → TWILIO ──────────────────────────────
  async function handleUserSpeech(text) {
    if (!text.trim() || isProcessing) return
    isProcessing = true
    console.log('User:', text)

    conversationHistory.push({ role: 'user', content: text })

    try {
      // 1. Get AI response from Groq
      const completion = await getGroq().chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        max_tokens: 120,
        temperature: 0.6,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          ...conversationHistory,
        ],
      })

      let aiText = completion.choices[0]?.message?.content || "I'm sorry, could you repeat that?"
      const transferRequested = aiText.includes('[TRANSFER_TO_HUMAN]')
      aiText = aiText.replace('[TRANSFER_TO_HUMAN]', '').trim()

      console.log('Aria:', aiText)
      conversationHistory.push({ role: 'assistant', content: aiText })

      // 2. Generate TTS audio from OpenAI
      const ttsResponse = await getOAI().audio.speech.create({
        model: 'tts-1',
        voice: 'nova',
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
    const greeting = "Hello, thank you for calling Bright Smile Dental — this is Aria, how can I help you today?"
    conversationHistory.push({ role: 'assistant', content: greeting })

    try {
      const ttsResponse = await getOAI().audio.speech.create({
        model: 'tts-1',
        voice: 'nova',
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
        dgLive = await startDeepgram()
        // Small delay to ensure Twilio stream is ready to receive audio
        setTimeout(() => sendGreeting(), 500)
        break

      case 'media':
        if (dgLive) {
          const audioData = Buffer.from(msg.media.payload, 'base64')
          dgLive.send(audioData)
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
