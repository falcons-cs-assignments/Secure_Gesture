from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# Get default audio device using PyCAW
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# Get current volume
currentVolumeDb = volume.GetMasterVolumeLevel()
volume.SetMasterVolumeLevel(currentVolumeDb - 6.0, None)
# NOTE: -6.0 dB = half volume !

#
# from __future__ import print_function
# from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
#
#
# def main():
#     sessions = AudioUtilities.GetAllSessions()
#     for session in sessions:
#         volume = session._ctl.QueryInterface(ISimpleAudioVolume)
#         if session.Process and session.Process.name() == "vlc.exe":
#             print("volume.GetMasterVolume(): %s" % volume.GetMasterVolume())
#             volume.SetMasterVolume(0.6, None)
#
#
# if __name__ == "__main__":
#     main()
