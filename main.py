import cv2
import os
import time

from selenium import webdriver

# SETTINGS
ALBUM_FOLDER = "album_covers"

MATCH_THRESHOLD = 25

# Put the FULL YouTube URL for each song here
songs = {
    "Currents": "https://youtu.be/c3yEjD_oijw?si=v9zGBNOh3lU99Ht4",
    "DoThatAgain": "https://youtu.be/3KQjOE1AuN4?si=iwvN_vdRvEWLB5JH",
    "UnVeranoSinTi" : "https://youtu.be/1ZJCDGUGc1o?si=LHQAjE8FSyD4DVFm",
    "JordanWard" : "https://youtu.be/c_hxN6eAros?si=1gfDWb_Xh33tCYvW",
    "RED" : "https://youtu.be/AgFeZr5ptV8?si=GBih-3-YfLbWqymm"
}
#--------------------------------------
# START CHROME
#--------------------------------------
print("Starting Chrome...")

driver = webdriver.Chrome()

# Open YouTube
driver.get("https://www.youtube.com")

print("Chrome started.")
#--------------------------------------
# SET UP ALBUM DETECTOR
# --------------------------------

orb = cv2.ORB_create(nfeatures=1500)

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING
)

albums = []
# --------------------------------
# LOAD ALBUM COVERS
# --------------------------------

for filename in os.listdir(ALBUM_FOLDER):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        path = os.path.join(
            ALBUM_FOLDER,
            filename
        )

        image = cv2.imread(path)

        if image is None:
            print("Could not load:", filename)
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        keypoints, descriptors = orb.detectAndCompute(
            gray,
            None
        )

        album_name = os.path.splitext(
            filename
        )[0]

        albums.append({
            "name": album_name,
            "descriptors": descriptors
        })

        print("Loaded:", album_name)


# --------------------------------
# START CAMERA
# --------------------------------

camera = cv2.VideoCapture(0)

current_album = None

# Album has to be detected several times
# before switching songs
candidate_album = None
candidate_frames = 0

CONFIRM_FRAMES = 5


# --------------------------------
# CAMERA LOOP
# --------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not access camera.")
        break


    # Convert camera image to grayscale
    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Find features in camera image
    frame_keypoints, frame_descriptors = orb.detectAndCompute(
        frame_gray,
        None
    )


    best_album = None
    best_matches = 0


    # --------------------------------
    # COMPARE EVERY ALBUM
    # --------------------------------

    if frame_descriptors is not None:

        for album in albums:

            if album["descriptors"] is None:
                continue


            matches = matcher.knnMatch(
                album["descriptors"],
                frame_descriptors,
                k=2
            )


            good_matches = []


            for match_pair in matches:

                if len(match_pair) == 2:

                    first, second = match_pair

                    if first.distance < 0.75 * second.distance:

                        good_matches.append(first)


            # Is this our best match?
            if len(good_matches) > best_matches:

                best_matches = len(good_matches)

                best_album = album["name"]


    # --------------------------------
    # SHOW MATCH COUNT
    # --------------------------------

    cv2.putText(
        frame,
        "Matches: " + str(best_matches),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    # --------------------------------
    # ALBUM DETECTED
    # --------------------------------

    if best_matches >= MATCH_THRESHOLD:

        cv2.putText(
            frame,
            best_album + " DETECTED!",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Check if we're seeing the same
        # candidate album repeatedly
        if best_album == candidate_album:

            candidate_frames += 1

        else:

            candidate_album = best_album
            candidate_frames = 1


        # --------------------------------
        # SWITCH SONG
        # --------------------------------

        if candidate_frames >= CONFIRM_FRAMES:

            if best_album != current_album:

                if best_album in songs:

                    print(
                        "Switching to:",
                        best_album
                    )

                    # Navigate SAME Chrome tab
                    # to the new YouTube video
                    driver.get(
                        songs[best_album]
                    )

                    current_album = best_album

                else:

                    print(
                        "No song found for:",
                        best_album
                    )


    else:

        candidate_album = None
        candidate_frames = 0

        cv2.putText(
            frame,
            "Show an album cover",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # --------------------------------
    # SHOW CAMERA
    # --------------------------------

    cv2.imshow(
        "Album Detector",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# CLEAN UP
# --------------------------------

camera.release()

cv2.destroyAllWindows()

driver.quit()