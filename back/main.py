import subprocess
import time

def launch_scripts_sequentially(scripts):
    """
    Launches a list of Python scripts sequentially.

    Args:
        scripts (list): A list of script filenames to execute.
    """
    print("Launching scripts sequentially...")
    for script in scripts:
        print(f"--- Starting {script} ---")
        try:
            # Using subprocess.run waits for each script to complete
            subprocess.run(["python", script], check=True)
            print(f"--- Finished {script} ---")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            # Decide if you want to stop or continue on error
            # break
        except FileNotFoundError:
            print(f"Error: {script} not found. Please ensure the script exists in the same directory.")
            # break
    print("All sequential scripts have been executed.")

def launch_scripts_concurrently(scripts):
    """
    Launches a list of Python scripts concurrently.

    Args:
        scripts (list): A list of script filenames to execute.
    """
    print("Launching scripts concurrently...")
    processes = []
    for script in scripts:
        try:
            # subprocess.Popen starts the script in a new process without blocking
            processes.append(subprocess.Popen(["python", script]))
            print(f"--- Launched {script} ---")
        except FileNotFoundError:
            print(f"Error: {script} not found. Please ensure the script exists in the same directory.")

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    print("All concurrent scripts have finished.")

if __name__ == "__main__":
    # List of scripts to be launched
    scripts_to_launch = [
        "master_coordinator.py",
        "inference_monitor.py",
        "prompt_generator.py"
    ]

    # To launch the scripts sequentially, uncomment the following line:
    # launch_scripts_sequentially(scripts_to_launch)

    # To launch the scripts concurrently, uncomment the following line:
    launch_scripts_concurrently(scripts_to_launch)

    # To demonstrate both, you can run them one after the other
    # print("\n" + "="*50 + "\n")
    # time.sleep(2)
    # launch_scripts_sequentially(scripts_to_launch)