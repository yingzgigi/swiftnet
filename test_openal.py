from openal import *
import time
import numpy as np
import struct
import ctypes
import itertools
import wave


"""
wavesound = oalOpen("sound/LiquidWater.wav")

wavesound.play()

while wavesound.get_state()==AL_PLAYING:
    time.sleep(1)
    
oalQuit()
"""





class Controller():
    def Source(self, sources):
        alGenSources(1, sources)
    
    def setPosition(self, sources, x, y, z, distance):
        alSource3f(sources, AL_POSITION, x, y, z)
        alSourcef(sources, AL_GAIN, 1/distance)
        
    def SourcePlay(self, sources, buffers):
        alSourcei(sources, AL_BUFFER, buffers)
        alSourcePlay(sources)
        
    def SourceDelete(self, sources):
        alDeleteSources(1, sources)
        
    def setLopping(self, sources, loop):
        alSourcei(sources, AL_LOOPING, AL_TRUE if loop else AL_FALSE)
        
    def isPlaying(self, sources):
        state = ALint(0)
        alGetSourcei(sources, AL_SOURCE_STATE, state)
        if state.value == AL_PLAYING:
            return True
        else:
            return False
        
    def pause(self, sources):
        alSourcePause(sources)
        
    def continuewPlaying(self, sources):
        alSourcePlay(sources)

class WaveStreamer(object):
    def __init__(self, filename):
        wav = wave.open(filename)
        self.__wav = wav
        self.__channels = wav.getnchannels()
        self.__bit_rate = wav.getsampwidth() * 8
        self.__sample_rate = wav.getframerate()
        self.__num_frames = wav.getnframes()
        self.__current_frame = 0

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.__wav is not None:
            self.__wav.close()
            self.__wav = None

    @property
    def channels(self):
        return self.__channels

    @property
    def bit_rate(self):
        return self.__bit_rate

    @property
    def sample_rate(self):
        return self.__sample_rate

    @property
    def num_frames(self):
        return self.__num_frames

    @property
    def current_frame(self):
        return self.__current_frame

    @property
    def duration(self):
        return self.num_frames / float(self.sample_rate)

    @property
    def format(self):
        fm = {
            (1, 8): al.AL_FORMAT_MONO8,
            (2, 8): al.AL_FORMAT_STEREO8,
            (1, 16): al.AL_FORMAT_MONO16,
            (2, 16): al.AL_FORMAT_STEREO16,
        }
        return fm[(self.channels, self.bit_rate)]

    def read_frames(self, n=1):
        assert n > 0
        assert self.current_frame < self.num_frames
        return self.__wav.readframes(n)


def load_wav(): #方案2 水流声
    wave_file = 'sound/LiquidWater.wav'
    
    wav = WaveStreamer(wave_file)
    data = wav.read_frames(wav.num_frames)

    # convert to float array for processing
    assert wav.bit_rate == 16
    print("num_frames:", len('%ih'%wav.num_frames))
    print("data:", len(data))
    #inputs = np.array(struct.unpack('%ih' % wav.num_frames, data)).astype(np.float32) #/ 32768.

    return wav.sample_rate, wav.format, data


class AudioMaster():

    def loadSound(self, file, buffers, buffers_list):
        
        alGenBuffers(1,buffers)
        
        wavefile = WaveStreamer(file)
        data = wavefile.read_frames(wavefile.num_frames)
        
        buffers_list.append(buffers)
        alBufferData(buffers, wavefile.format, data, wavefile.num_frames, wavefile.sample_rate)
        wavefile.close()
        return buffers
    
    def setListenerData(self, u, v, w):
        alListener3f(AL_POSITION, u, v, w)
        alListener3f(AL_VELOCITY, 0, 0, 0)
        
    def cleanUp(buffers, buffer_list):
        for buffers in buffer_list:
            alDeleteBuffers(1, buffers)
    
        


    
    
"""
SourceLoader("sound/LiquidWater.wav")

device = alcOpenDevice()
Source()
alcCloseDevice(device)
"""

device = alcOpenDevice(None)

context = alcCreateContext(device, None)
alcMakeContextCurrent(context)

x = 4
y = 5
z = -3
u = 10
v = 10
w = 10
distance = np.sqrt(np.square(x) + np.square(y) + np.square(z))

#方案1
test = AudioMaster()
test.setListenerData(u,v,w)

buffers_list = []
buffers = ALuint()
#tbuffer = 0

source = ALuint()
tbuffer = test.loadSound("sound/Footstep.wav", buffers, buffers_list)
controller = Controller()
controller.Source(source)
controller.setLopping(source, True)
controller.SourcePlay(source, AL_TRUE) ###############
controller.setPosition(source, x, y, z, distance)

'''
rate, formats, data = load_wav()
sample_len = len(data)
channels = data
#audio buffer
buffers = ALuint()
alGenBuffers(1, buffers)
alBufferData(buffers, formats, channels, sample_len, rate)


#controller
sources = ALuint()
controller = Controller()
controller.Source(sources)



# binding the source to the buffer
alSourceQueueBuffers(sources, 1, buffers)
#print(buffers)

alSourcePlay(sources)
#controller.SourcePlay(buffers)
'''
while True:
    #if controller.isPlaying(source):
    x+=10
    distance = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    controller.setPosition(source, x, y, z, distance)
    time.sleep(10)
        #controller.pause(source)
    #else:
        #controller.continuewPlaying(source)
'''
state = ALint(0)
while True:
    alGetSourcei(source, AL_SOURCE_STATE, state)
    if state.value != AL_PLAYING:
        #test()
        break
'''   
'''
while wavesound.get_state()==AL_PLAYING:
    time.sleep(1)
'''
#alSourcei(source, AL_BUFFER, 0)
controller.SourceDelete(source)
AudioMaster.cleanUp(buffers, buffers_list)
#clean up
#alDeleteBuffers(1, buffers)
#alDeleteSources(1, source)
alcDestroyContext(context)
alcCloseDevice(device)