import os


def delete_non_hpge_detectors(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    count = 0

    # Iterate through all .spe files
    for filename in os.listdir(directory):
        if filename.endswith('.spe'):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        if line.startswith('Detector:'):
                            detector_info = line.split('Detector:', 1)[1].strip()

                            # If it doesn't start with HPGe, delete the file
                            if not detector_info.startswith('HPGe'):
                                # Close the file before deleting it to avoid permission errors on some OS
                                file.close()
                                os.remove(filepath)
                                print(f"Deleted {filename} - {detector_info}")
                                count += 1
                            break
            except Exception as e:
                print(f"Could not process {filename}: {e}")

    print(f"\nSuccessfully deleted {count} files.")


if __name__ == "__main__":
    # Define your directory path here
    target_directory = 'data/IAEA_HPGe/2023/spe'
    delete_non_hpge_detectors(target_directory)