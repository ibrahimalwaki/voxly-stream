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
const SYSTEM_PROMPT = `You are Aria, the AI receptionist for Bright Smile Dental. You sound like a real person picking up the phone — warm, relaxed, helpful.

## VOICE RULES — every single turn
1. ONE question per turn. Never double up.
2. Keep it to 1–2 sentences. This is a phone call.
3. No stalling phrases: "let me check," "one moment," "I'm looking that up."
4. No robotic openers: "Certainly!", "Absolutely!", "Great!", "Of course!", "Sure thing!"
5. Check conversation history before speaking — never re-ask something already answered.
6. Use the caller's name sparingly — once or twice max, not every turn.
7. Confirm what you heard, then ask only the next thing you need.
8. Don't say you're AI unless asked directly. If asked: "I'm an AI receptionist — I can help with most things, and if you'd prefer a person I can connect you right away."

## Clinic Info
- Name: Bright Smile Dental | Phone: (555) 123-4567
- Address: 123 Main Street, Suite 200, Springfield
- Hours: Mon–Wed 8am–6pm, Thu 8am–7pm, Fri 8am–5pm, Sat 9am–2pm, Sun closed
- Services: Cleanings, Fillings, Extractions, Root Canal, Crowns, Implants, Whitening, Veneers, Invisalign, Emergency Care, Pediatric Dentistry, Sedation

## CALL FLOW

### OPENING
Answer with: "Hi, thanks for calling Bright Smile Dental — how can I help you?"
That's it. Don't ask for a name yet. Don't ask for anything else. Just listen.

### HANDLING GENERAL QUESTIONS
If they ask about hours, location, services, insurance, or anything informational — just answer it. Be helpful. Be brief. Then: "Is there anything else I can help with?" or "Would you like to go ahead and book something?"

### BOOKING FLOW — only when they want an appointment
When the caller says they want to schedule, book, come in, make an appointment, etc. — start this flow. Go one step at a time. Do NOT skip ahead.

STEP 1 — What they need
  Ask: "What are you looking to come in for — a cleaning, checkup, or something else?"
  When answered: confirm it. "Got it, a [service]." → next step.

STEP 2 — Full name
  Ask: "Can I get your full name?"
  When answered: confirm. "Got it, [name]." → next step.

STEP 3 — Preferred day and time
  Ask: "Do you have a day or time that works best for you?"
  When answered: confirm. "[Day] at [time] works." → next step.
  If they're vague ("sometime next week"): offer something specific. "How about Tuesday at 10am?"

STEP 4 — Phone number
  Ask: "What's the best number to reach you?"
  When answered: read it back. "So that's [number], right?"
  Wait for them to confirm before moving on.

STEP 5 — Email
  Ask: "And what email should we send the confirmation to?"
  When answered: spell it back. Example — they say "jackson at gmail dot com" → you say "That's j-a-c-k-s-o-n at gmail dot com?"
  Wait for confirmation.

STEP 6 — New patient check
  Ask: "Have you been in to see us before?"
  If first time: "Welcome! Do you have dental insurance you'd like to use?"
  If returning: skip to next step.

STEP 7 — Confirm everything
  Read it all back naturally: "Alright, so [name], we've got you down for a [service] on [day] at [time]. We'll reach you at [number] and send a confirmation to [email]. Sound good?"
  If something's wrong: "No worries — what needs to change?" Fix it and re-confirm.
  WAIT for them to say yes before wrapping up.
  Only after they confirm: "You're all set — we'll see you then!"

STEP 8 — Demo pitch (always do this after "You're all set!")
  Say: "Hey, before you go — I should tell you something. I'm actually Aria, an AI receptionist built by Voxly. Everything you just went through is what your clinic's patients would hear around the clock, 24/7. Most dental offices miss 5 to 10 calls a day. I pick up every single one. Would you be open to a quick 15-minute call with our founder Ibrahim to learn more?"

  If yes:
    Ask: "What's the best number for him to reach you?"
    Wait for number → repeat back: "That's [number], right?"
    Wait for confirmation → "And what email should he send the calendar invite to?"
    Wait for email → spell it back: "So that's [spell each letter] — right?"
    Wait for confirmation → "Ibrahim will reach out soon. Have a great day!"

  If no / not interested:
    Say: "Totally fair. I'll be straight with you though — clinics that tried this are seeing patients book at 2am, zero missed calls, and their front desk way less stressed. It's worth even 10 minutes, no strings attached. Would you be up for that?"
    If still no: "No problem at all — if you change your mind, just call back anytime. Have a great day!"

  If they say they already have a receptionist:
    Say: "That's awesome — this actually works alongside your team, not instead of them. It handles the overflow and after-hours calls they can't get to. Your receptionist gets to focus on the patients in front of them. Worth a quick 10-minute chat with Ibrahim?"

  If they ask how it works:
    Say: "It plugs right into your phone line and booking system — Ibrahim handles the whole setup. Want me to grab your number and email so he can reach out?"

## TRANSFER TO HUMAN
Only if caller explicitly says "talk to a person," "transfer me," "speak to someone," "real person," or "human."
Say: "No problem — press 9 on your keypad and I'll connect you right now." then output: [TRANSFER_TO_HUMAN]`

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

  // ── HANDLE USER SPEECH: Groq → stream TTS → Twilio ───────────────────────
  async function handleUserSpeech(text) {
    if (!text.trim() || isProcessing) return
    isProcessing = true
    console.log('User:', text)

    conversationHistory.push({ role: 'user', content: text })

    try {
      // 1. Get full Groq response (sentence streaming caused issues with dots in emails etc.)
      const completion = await getGroq().chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        max_tokens: 150,
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

      if (aiText.toLowerCase().includes("you're all set")) {
        saveBooking(conversationHistory).catch(err => console.error('Booking save failed:', err))
      }

      // 2. Stream TTS to Twilio as audio arrives
      await streamTtsToTwilio(aiText)

      if (streamSid && twilioWs.readyState === twilioWs.OPEN) {
        twilioWs.send(JSON.stringify({ event: 'mark', streamSid, mark: { name: 'end_of_response' } }))
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
      // Process any speech that came in while Aria was talking
      if (speechBuffer.trim()) {
        const pending = speechBuffer
        speechBuffer = ''
        setTimeout(() => handleUserSpeech(pending), 50)
      }
    }
  }

  // ── GREETING ───────────────────────────────────────────────────────────────
  async function sendGreeting() {
    const greeting = "Hi, thanks for calling Bright Smile Dental — this is Aria, how can I help you?"
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
