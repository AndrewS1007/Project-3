# AGENTS.md

## Project

Python 3 application for 3D hand sensing and computer navigation.
The program captures camera input to track hands and will eventually map gestures to controls, such as mouse or keyboard.
Currently uses MediaPipe for hand tracking, OpenCV for video capture, and Pygame for rendering the HUD and skeleton.

# Install dependencies

pip install opencv-python mediapipe pygame

# Run the main application

python demo.py

## Conventions

* Four-space indent, no tabs.

* Use `snake_case` for variables and functions, and `PascalCase` for classes.

* Add type hints to new function signatures.

* **Dependency Policy**: Do not introduce any new third-party dependencies without asking a human first. All suggested packages must be verified against PyPI to avoid hallucinated packages.

## Do not

* Do not edit the camera initialization backend as this is specifically configured for Windows Compatability

* Do not attempt to bypass or mock the camera input loop without explicit instructions.
