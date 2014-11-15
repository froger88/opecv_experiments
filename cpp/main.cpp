/*
code based on http://docs.opencv.org/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html

CHANGES:
    1. detecting face only in region of upperboddy
    2. print rectangles instead of circles/elipses

TODO: (want to do):
    1. new upperbody might not be detected inside of existing upperbody
    2. simple identify upperbodies (by position):
        2.1 save time of upperbody detected
        2.2 save time of face inside of upperbody detected
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "../haar_files/haarcascade_frontalface_alt.xml";
String hs_cascade = "../haar_files/HS.xml";
CascadeClassifier face_cascade;
CascadeClassifier hs;
string window_name = "Capture - Face detection";

/** @function main */
int main(int argc, const char **argv) {
    CvCapture *capture;
    Mat frame;

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };
    if (!hs.load(hs_cascade)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    //-- 2. Read the video stream
    capture = cvCaptureFromCAM(0);
    if (capture) {
        while (true) {
            frame = cvQueryFrame(capture);

            //-- 3. Apply the classifier to the frame
            if (!frame.empty()) {detectAndDisplay(frame);}
            else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            int c = waitKey(10);
            if ((char) c == 'q') {break;}
        }
    }
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame) {
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces

    std::vector<Rect> upperbody;
    //-- In each face, detect upperbody
    hs.detectMultiScale(frame, upperbody, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (size_t j = 0; j < upperbody.size(); j++) {
        rectangle(frame, //img
                Point(upperbody[j].x, upperbody[j].y), //pt1
                Point(upperbody[j].x+upperbody[j].width,upperbody[j].y+upperbody[j].height), //pt2
                Scalar(0, 0, 255), //color
                4, //thickness
                8, //lineType
                0); //shift

        Mat x = frame_gray(upperbody[j]);
        face_cascade.detectMultiScale(x, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, //img
                    Point(upperbody[j].x + faces[i].x, upperbody[j].y +faces[i].y), //pt1
                    Point(upperbody[j].x + faces[i].x+faces[i].width,upperbody[j].y + faces[i].y+faces[i].height), //pt2
                    Scalar(0, 255, 0), //color
                    4, //thickness
                    8, //lineType
                    0); //shift
        }
    }
    //-- Show what you got
    imshow(window_name, frame);
}
