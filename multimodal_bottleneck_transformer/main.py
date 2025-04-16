# Import relevant libraries
import datetime
from utils.global_vars import parameters

# Import useer defined library
from utils.mbt_runner_class import MBTRunnerClass


if __name__ == '__main__':
    print("Starting run for MBT module at {}".format(datetime.datetime.now()))

    # Initialise the parameters for the model
    parameters = parameters.copy()
    parameters["dataset"] = "AudioSet"  # 'AudioSet', 'vggsound'
    parameters["top_k_labels"] = 3
    parameters["modality"] = 'av'  # 'a', 'v', 'av'
    parameters["epochs"] = 2
    parameters["vgg_sound_lr"] = 0.01
    parameters["audio_set_lr"] = 1e-4
    parameters["batch_size"] = 2

    # Initialise MBT runner model class
    runner = MBTRunnerClass(parameters)
    runner.load_format_data()
