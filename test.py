"""Plays the choppin minigame."""

import logging

import psutil

from autocv import AutoCV

logging.basicConfig(
    level="INFO",
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_process_dlls(process_name):
    """ List DLLs loaded by processes matching 'process_name'. """
    # Iterate over all processes
    for proc in psutil.process_iter(['pid', 'name']):
        # Check if this process has the specified name
        if process_name.lower() in proc.info['name'].lower():
            # Get the process object using PID
            p = psutil.Process(proc.info['pid'])
            # Print process information
            print(f"Process ID: {p.pid}, Process Name: {p.name()}")
            # Iterate through DLLs this process has loaded
            for dll in p.memory_maps():
                if 'dll' in dll.path.lower():
                    print(dll.path)


def start():
    logging.info("Initializing...")
    _autocv = AutoCV()
    _autocv.set_hwnd_by_title("Legends Of Idleon")
    # _autocv.set_inner_hwnd_by_title("Chrome_RenderWidgetHostHWND")
    print(_autocv.get_child_windows())
    return _autocv


def main() -> None:
    # find_process_dlls("legendsofidleon.exe")
    autocv = start()

    while True:
        autocv.refresh()
        autocv.show_backbuffer(live=True)


if __name__ == "__main__":
    main()
