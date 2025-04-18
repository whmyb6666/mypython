import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output2.wav"

timeDealy  =5

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

totalbuffer = int(RATE / CHUNK *timeDealy)

for i in range(0,int(RATE / CHUNK * RECORD_SECONDS)):
    print('i',i)
    data = stream.read(CHUNK)
    print(data)
    frames.append(data)
    if len(frames) > totalbuffer:
        print(f"delete frame {len(frames)}")
        del frames[0]

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
