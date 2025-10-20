import { useState, useEffect, useRef } from "react";
import { Send, Loader2 } from "lucide-react";
import { grpcClient } from "../services/grpcClient";
import { Message } from "../types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatInterfaceProps {
  videoId?: string;
  isConnected: boolean;
  initialMessages?: Message[];
}

export default function ChatInterface({ videoId, isConnected, initialMessages = [] }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Create a unique key from initialMessages to detect changes
  const initialMessagesRef = useRef<string>("");
  const currentKey = JSON.stringify(initialMessages.map(m => ({ id: m.id, text: m.text?.substring(0, 50) })));
  
  useEffect(() => {
    // Only update if the content actually changed
    if (currentKey !== initialMessagesRef.current) {
      console.log("💬 ChatInterface: initialMessages changed!");
      console.log("💬 New messages count:", initialMessages.length);
      console.log("💬 First message:", initialMessages[0]?.text?.substring(0, 50));
      
      // Completely replace the messages state
      setMessages(initialMessages);
      initialMessagesRef.current = currentKey;
    }
  }, [initialMessages, currentKey]);

  useEffect(() => {
    // Send welcome message when video is loaded
    if (videoId) {
      addSystemMessage("Video loaded! You can now ask questions about the video.");
    }
  }, [videoId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const addSystemMessage = (text: string) => {
    const message: Message = {
      id: `msg_${Date.now()}`,
      text,
      sender: "system",
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, message]);
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading || !isConnected) return;

    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      text: input,
      sender: "user",
      timestamp: Date.now(),
    };

    // Check if this is the first user message (auto-rename session)
    const isFirstMessage = messages.length === 0 || 
      messages.filter(m => m.sender === "user").length === 0;

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput("");
    setIsLoading(true);

    try {
      // Auto-rename session based on first message
      if (isFirstMessage && currentInput.trim()) {
        const sessionId = grpcClient.getSessionId();
        // Generate a title from first message (first 40 chars or first sentence)
        let title = currentInput.trim();
        const firstSentence = title.split(/[.!?]/)[0];
        title = firstSentence.length > 0 && firstSentence.length < 60 
          ? firstSentence 
          : title.substring(0, 40);
        if (title.length < currentInput.length) {
          title += "...";
        }
        
        console.log("🏷️ Auto-renaming session to:", title);
        await grpcClient.updateSession(sessionId, title);
      }

      // Stream responses from backend
      for await (const response of grpcClient.chat(currentInput, videoId)) {
        setMessages((prev) => [...prev, response]);
      }
    } catch (error) {
      console.error("Chat error:", error);
      addSystemMessage(`Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full w-full overflow-hidden">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-w-0">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">Welcome to Video Analysis AI</h2>
              <p className="text-sm">
                {videoId
                  ? "Ask me anything about your video!"
                  : "Upload a video to get started"}
              </p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"} w-full min-w-0`}
          >
            <div
              className={`max-w-[70%] min-w-0 rounded-lg p-3 break-words ${
                message.sender === "user"
                  ? "bg-blue-600 text-white"
                  : message.sender === "assistant"
                  ? "bg-gray-700 text-white"
                  : "bg-gray-800 text-gray-300 text-sm"
              }`}
            >
              {message.agentInfo && (
                <div className="text-xs opacity-75 mb-1 break-words">
                  🤖 {message.agentInfo.agentName}: {message.agentInfo.action}
                </div>
              )}
              <div className="prose prose-invert max-w-none prose-sm break-words overflow-wrap-anywhere">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.text}
                </ReactMarkdown>
              </div>
              <div className="text-xs opacity-50 mt-1 break-words">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-700 rounded-lg p-3 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-700 p-4 bg-gray-800 flex-shrink-0">
        <div className="flex gap-2 w-full min-w-0">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              isConnected
                ? "Type your message..."
                : "Connecting to backend..."
            }
            disabled={!isConnected || isLoading}
            className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading || !isConnected}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg px-4 py-2 flex items-center gap-2 transition-colors"
          >
            <Send className="w-4 h-4" />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
