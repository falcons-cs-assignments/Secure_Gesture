import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class Speaker:
    def __init__(self, max_distance=0.55):
        """
            distance: value in range [0.0 : 0.55]
            volume level: value in (dB) {0% --> -65.25dB,
                                        100% --> 0dB}
                                        # NOTE: -6.0 dB = half volume !
        """
        # Get default audio device using PyCAW
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

        self.distance_range = [0, max_distance]
        self.vol_range_dB = self.volume.GetVolumeRange()[:2]     # [-65.25, 0.0]
        # print(self.vol_range_dB)

    def __call__(self, distance):
        # map distance into volume level
        # TODO: make mapping is logarithmic as it for dB :)
        distance = min(distance, 0.55)
        ratio = distance / self.distance_range[1]
        level_dB = 28 * math.log(max(0.0001, ratio))   # level = ln(ratio)

        # Get current volume
        self.volume.SetMasterVolumeLevel(max(level_dB, self.vol_range_dB[0]), None)
        current_volume_dB = self.volume.GetMasterVolumeLevel()

        cur_vol = math.exp(current_volume_dB / 15) * 100
        return cur_vol


def main():
    volume = Speaker()
    print(volume(0.1))


if __name__ == "__main__":
    main()
