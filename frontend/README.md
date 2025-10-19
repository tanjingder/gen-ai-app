# Frontend Setup Guide

## Prerequisites

- Node.js 18+ and npm
- Rust (for Tauri)

## Installation

```powershell
# Install dependencies
npm install

# Install Tauri CLI globally (optional but recommended)
npm install -g @tauri-apps/cli
```

## Development

```powershell
# Run in development mode with hot reload
npm run tauri:dev
```

This will:
1. Start the Vite dev server
2. Launch the Tauri application
3. Enable hot reload for quick development

## Building

```powershell
# Build for production
npm run tauri:build
```

The built application will be in `src-tauri/target/release/`.

## Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   │   ├── ChatInterface.tsx
│   │   ├── VideoUpload.tsx
│   │   └── Header.tsx
│   ├── services/         # gRPC client
│   │   └── grpcClient.ts
│   ├── types/            # TypeScript types
│   │   └── index.ts
│   ├── App.tsx          # Main app component
│   ├── main.tsx         # Entry point
│   └── index.css        # Global styles
├── src-tauri/           # Tauri backend (Rust)
└── package.json
```

## gRPC Integration

The frontend communicates with the Python backend via gRPC. The gRPC client is implemented in `src/services/grpcClient.ts`.

### Current Implementation

Currently using a mock implementation. To enable real gRPC:

1. Generate TypeScript types from proto files:
```powershell
npm run proto:generate
```

2. Implement Tauri commands in `src-tauri/src/main.rs` to handle gRPC calls

3. Update `grpcClient.ts` to use Tauri invoke API

## Styling

Uses Tailwind CSS for styling. Configuration in:
- `tailwind.config.js`
- `postcss.config.js`

## Features

- **Chat Interface**: Conversational UI for interacting with AI
- **Video Upload**: Drag-and-drop video upload
- **Real-time Responses**: Streaming responses from backend
- **Agent Visibility**: Shows which agents are being used
- **Connection Status**: Displays backend connection status

## TODO

- [ ] Integrate real gRPC client
- [ ] Add Tauri invoke commands for backend communication
- [ ] Implement file download for generated reports
- [ ] Add video player preview
- [ ] Implement analysis status tracking
- [ ] Add settings panel
- [ ] Implement session persistence
