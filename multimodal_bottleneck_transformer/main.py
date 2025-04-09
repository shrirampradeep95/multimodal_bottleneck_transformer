# Import relevant libraries
import datetime
from utils.global_vars import parameters

# Import useer defined library
from utils.mbt_runner_class import MBTRunnerClass


if __name__ == '__main__':
    print("Starting run for MBT module at {}".format(datetime.datetime.now()))

    # Initialise the parameters for the model
    parameters = parameters.copy()
    parameters["dataset"] = "vggsound"  # 'AudioSet', 'vggsound'
    parameters["videos_to_use"] = 10
    parameters["modality"] = 'av'  # 'a', 'v', 'av'
    parameters["epochs"] = 50
    parameters["vgg_sound_lr"] = 0.01
    parameters["audio_set_lr"] = 1e-4
    parameters["batch_size"] = 1

    # Initialise MBT runner model class
    runner = MBTRunnerClass(parameters)
    runner.load_format_data()
