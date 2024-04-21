import React, { useEffect, useRef } from "react";
import "./video.css";

const Video = () => {
  console.log("video");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    // Request access to the webcam
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          video.play().catch((error) => {
            console.error("Error playing the video stream", error);
          });
        }
      })
      .catch((error) => {
        console.error("Error accessing the webcam", error);
      });
  }, []);

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current || document.createElement("canvas");
    canvasRef.current = canvas;

    if (video && canvas) {
      // Set canvas size to video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/png"); // Converts the canvas content to a base64 URL
      }
    }
    return null;
  };

  return (
    <div className="form-container">
      <video ref={videoRef} id="video" className="vid" autoPlay></video>
      <button id="capture" onClick={capturePhoto}>
        Capture!!
      </button>
    </div>
  );
};

export default Video;
