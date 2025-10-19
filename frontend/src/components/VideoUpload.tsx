import { useState, useRef } from "react";
import { Upload, X, CheckCircle } from "lucide-react";
import { grpcClient } from "../services/grpcClient";
import { VideoInfo } from "../types";

interface VideoUploadProps {
  onVideoUploaded: (videoInfo: VideoInfo) => void;
  currentVideo?: VideoInfo | null;
}

export default function VideoUpload({ onVideoUploaded, currentVideo }: VideoUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    // Don't allow drag-and-drop if video already uploaded
    if (currentVideo) return;
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    // Don't allow drop if video already uploaded
    if (currentVideo) return;

    const files = Array.from(e.dataTransfer.files);
    const videoFile = files.find((file) =>
      file.type.startsWith("video/")
    );

    if (videoFile) {
      await uploadFile(videoFile);
    } else {
      alert("Please upload a video file");
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadFile(file);
    }
  };

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      const videoInfo = await grpcClient.uploadVideo(file);

      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadedFile(file.name);
      
      setTimeout(() => {
        onVideoUploaded(videoInfo);
        setIsUploading(false);
        setUploadProgress(0);
      }, 500);
    } catch (error) {
      console.error("Upload failed:", error);
      alert(`Upload failed: ${error}`);
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleClearUpload = () => {
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="w-full">
      <h2 className="text-lg font-semibold mb-4">Upload Video</h2>

      {!uploadedFile ? (
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            currentVideo
              ? "border-gray-700 bg-gray-800/50 cursor-not-allowed"
              : isDragging
              ? "border-blue-500 bg-blue-500/10"
              : "border-gray-600 hover:border-gray-500"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading || currentVideo !== null}
          />

          {!isUploading ? (
            <>
              <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-gray-300 mb-2">
                Drag and drop video file here
              </p>
              <p className="text-sm text-gray-400 mb-4">or</p>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={currentVideo !== null}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  currentVideo
                    ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                Browse Files
              </button>
              <p className="text-xs text-gray-500 mt-4">
                Supports: MP4, AVI, MOV, MKV (max 500MB)
              </p>
            </>
          ) : (
            <div>
              <div className="w-12 h-12 mx-auto mb-4 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-gray-300 mb-2">Uploading...</p>
              <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-400">{uploadProgress}%</p>
            </div>
          )}
        </div>
      ) : (
        <div className="border border-green-500 rounded-lg p-4 bg-green-500/10">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
              <div>
                <p className="text-green-400 font-medium">Video Uploaded</p>
                <p className="text-sm text-gray-300 truncate">{uploadedFile}</p>
              </div>
            </div>
            <button
              onClick={handleClearUpload}
              className="text-gray-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
