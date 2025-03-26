# Baseball-Cinematography
In this project, I leverage computer vision models found on BaseballCV GitHub (https://github.com/dylandru/BaseballCV) in order to track pitcher mechanics. With these pitcher mechanics, I utilized a method of time series analysis known as Dynamic Time Warping in order to compare the mechanics of different pitchers.

# Literature:
- Adler, D. (2024). Welcome to the world of MLB Gameday 3D.​
- Drummey, D. (2024). Utilizing single-angle broadcast feeds and computer vision to extract 3D MLB biomechanical data.​
- Jiang, T., et. al. (2023). RTMPose: Real-time multi-person pose estimation based on MMPose.​
- McElroy, L. (2022). Computer vision in baseball: The evolution of Statcast.​
- Redmon, J., et. al. (2015). You only look once: Unified, real-time object detection.​
- Taruskin, T. (2024). A visual scouting primer.​
- Wang, K., & Gasser, T. (1997). Alignment of curves by dynamic time warping.

# Notes:
scrape_pitch_ids.R will get the play IDs needed for scraping the mp4 files of a given pitch
tracking_and_analysis.py will download the broadcast video from Baseball Savant using the pitch IDs and will run the computer vision models/dynamic time warping analysis
