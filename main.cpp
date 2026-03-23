#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Load face cascade
CascadeClassifier faceCascade;

// Face detection
int detectFaces(Mat& frame) {
    vector<Rect> faces;
    Mat gray;

    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    faceCascade.detectMultiScale(gray, faces, 1.1, 4);
    return faces.size();
}

// Motion detection
double detectMotion(Mat& prev, Mat& curr) {
    Mat gray1, gray2, diff;

    cvtColor(prev, gray1, COLOR_BGR2GRAY);
    cvtColor(curr, gray2, COLOR_BGR2GRAY);

    absdiff(gray1, gray2, diff);

    return sum(diff)[0];
}

// Scoreboard region (adjust if needed)
Rect scoreboardROI(30, 30, 400, 120);

// Scoreboard change detection
double detectScoreboardChange(Mat& prev, Mat& curr) {
    Mat prevROI = prev(scoreboardROI);
    Mat currROI = curr(scoreboardROI);

    Mat gray1, gray2, diff;

    cvtColor(prevROI, gray1, COLOR_BGR2GRAY);
    cvtColor(currROI, gray2, COLOR_BGR2GRAY);

    absdiff(gray1, gray2, diff);

    return sum(diff)[0];
}

// Event structure
struct Event {
    int frameNumber;
    double score;
};

vector<Event> events;

int main() {

    // Load face model
    if (!faceCascade.load("models/haarcascade_frontalface_default.xml")) {
        cout << "Error loading face cascade\n";
        return -1;
    }

    // Open video
    VideoCapture cap("data/sample_video.mp4");

    if (!cap.isOpened()) {
        cout << "Error opening video\n";
        return -1;
    }

    Mat prev, curr;
    int frameNo = 0;
    int prevFaces = 0;

    while (cap.read(curr)) {

        // Skip frames for speed
        if (frameNo % 5 != 0) {
            frameNo++;
            continue;
        }

        if (!prev.empty()) {

            // Motion
            double motion = detectMotion(prev, curr);

            // Face detection
            int currentFaces = detectFaces(curr);

            double faceScore = 10;
            if (currentFaces > 10) faceScore = 100;
            else if (currentFaces > 5) faceScore = 60;
            else if (currentFaces > prevFaces + 3) faceScore = 80;

            // Scoreboard change
            double scoreChange = detectScoreboardChange(prev, curr);

            double scoreboardScore = 5;
            if (scoreChange > 800000) scoreboardScore = 120;
            else if (scoreChange > 300000) scoreboardScore = 80;

            // Motion score
            double motionScore = motion / 10000;

            // Final score
            double totalScore =
                0.4 * motionScore +
                0.3 * faceScore +
                0.3 * scoreboardScore;

            // Save event if important
            if (totalScore > 50) {
                events.push_back({frameNo, totalScore});
                cout << "Event detected at frame: " << frameNo << endl;
            }

            prevFaces = currentFaces;
        }

        prev = curr.clone();
        frameNo++;
    }

    cout << "\nTotal Events Detected: " << events.size() << endl;

    return 0;
}