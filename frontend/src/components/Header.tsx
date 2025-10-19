import { Circle, CheckCircle, Video } from "lucide-react";
import { VideoInfo } from "../types";

interface HeaderProps {
  isConnected: boolean;
  currentVideo: VideoInfo | null;
}

export default function Header({ isConnected, currentVideo }: HeaderProps) {
  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Video className="w-6 h-6 text-blue-500" />
          <h1 className="text-xl font-bold">Video Analysis AI</h1>
        </div>

        <div className="flex items-center gap-6">
          {currentVideo && (
            <div className="flex items-center gap-2 text-sm text-gray-300">
              <Video className="w-4 h-4" />
              <span className="hidden md:inline">{currentVideo.filename}</span>
            </div>
          )}

          <div className="flex items-center gap-2">
            {isConnected ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-500">Connected</span>
              </>
            ) : (
              <>
                <Circle className="w-4 h-4 text-red-500 animate-pulse" />
                <span className="text-sm text-red-500">Disconnected</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
