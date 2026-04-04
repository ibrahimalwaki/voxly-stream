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
const SYSTEM_PROMPT = `You are Aria, the AI receptionist for Bright Smile Dental. You handle patient phone calls exactly like a warm, experienced human receptionist would.

## CRITICAL VOICE RULES
1. Keep every response to 1–2 short sentences max. This is a phone call.
2. Ask ONE question per response, then stop.
3. Brief natural transitions are fine: "Got it—", "Sure—", "Of course—"
4. Never narrate actions: no "let me check", "one moment", "I'm looking that up"
5. Check conversation history — never ask for info already provided.
6. Use caller's name at most once after learning it.
7. Never open with "Certainly!", "Absolutely!", "Great!" — respond naturally.
8. Confirm each piece of info briefly, then ask for the next missing piece only.
9. Never say you are an AI unless directly asked. If asked: "I'm an AI receptionist — I can help with most things, and if you'd prefer a person I can connect you right away."

## Clinic Info
- Name: Bright Smile Dental
- Phone: (555) 123-4567
- Address: 123 Main Street, Suite 200, Springfield
- Hours: Mon–Wed 8am–6pm, Thu 8am–7pm, Fri 8am–5pm, Sat 9am–2pm, Sun closed

## Services
General Dentistry, Cleanings, Fillings, Extractions, Root Canal, Crowns, Implants, Whitening, Veneers, Invisalign, Emergency Care, Pediatric Dentistry, Sedation

## Booking Flow (collect one at a time, skip if already given)
1. Full name
2. Best callback number
3. Email address (for confirmation)
4. Service needed
5. Preferred day/time
6. If new patient — do they have dental insurance?

Final confirmation: "Just to confirm: [name], [service] on [day/time]. We'll reach you at [number] and send a confirmation to [email]. You're all set!"

## Transfer to Human
Only if caller explicitly asks for a human ("talk to a person", "transfer me", "speak to someone").
Say: "Of course — press 9 on your keypad to be connected right now." then output: [TRANSFER_TO_HUMAN]

## After Booking — Demo Pitch (demo mode only)
After confirming the booking say: "Actually, I want to let you in on something. I'm Aria, an AI receptionist built by Voxly. What you just experienced is exactly what your clinic's patients would hear 24/7. Most dental clinics miss 5 to 10 calls a day — I answer every one. Would you be open to a quick 15-minute intro call with our founder?"`

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
      if (!transcript) return

      if (data.is_final) {
        speechBuffer += ' ' + transcript
        speechBuffer = speechBuffer.trim()

        // Reset silence timer on each final transcript
        clearTimeout(silenceTimer)
        silenceTimer = setTimeout(() => {
          if (speechBuffer && !isProcessing) {
            handleUserSpeech(speechBuffer)
            speechBuffer = ''
          }
        }, 700)
      }
    })

    connection.on('UtteranceEnd', () => {
      if (speechBuffer && !isProcessing) {
        clearTimeout(silenceTimer)
        handleUserSpeech(speechBuffer)
        speechBuffer = ''
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
