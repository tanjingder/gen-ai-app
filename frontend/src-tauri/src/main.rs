// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use tonic::transport::Channel;

// Include generated gRPC code
pub mod video_analysis {
    tonic::include_proto!("video_analysis");
}

use video_analysis::{
    video_analysis_service_client::VideoAnalysisServiceClient,
    VideoChunk,
};

// Frontend response types (for Tauri invoke)
#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
    text: String,
    sender: String,
    timestamp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_used: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VideoMetadata {
    filename: String,
    duration_ms: i64,
    width: i32,
    height: i32,
    fps: f32,
    file_size: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct UploadResponse {
    success: bool,
    video_id: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<VideoMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReportResponse {
    success: bool,
    message: String,
    report_data: Vec<u8>,
    filename: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnalysisStatusResponse {
    video_id: String,
    stage: String,
    progress: f32,
    message: String,
}

// Session management response types
#[derive(Debug, Serialize, Deserialize)]
struct SessionResponse {
    session_id: String,
    title: String,
    created_at: String,
    updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    video_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    video_filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    video_metadata: Option<VideoMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionListResponse {
    sessions: Vec<SessionResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessageData {
    role: String,
    content: String,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionDataResponse {
    session: SessionResponse,
    messages: Vec<ChatMessageData>,
    has_cache: bool,
    cache_summary: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeleteSessionResponse {
    success: bool,
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct UpdateSessionResponse {
    success: bool,
    message: String,
}

/// Create gRPC client connection
async fn create_grpc_client() -> Result<VideoAnalysisServiceClient<Channel>, String> {
    match VideoAnalysisServiceClient::connect("http://localhost:50051").await {
        Ok(client) => Ok(client),
        Err(e) => Err(format!("Failed to connect to gRPC server on port 50051: {}", e)),
    }
}

/// Tauri command to send chat message to backend via gRPC
#[tauri::command]
async fn grpc_chat(
    message: String,
    session_id: Option<String>,
    video_id: Option<String>,
) -> Result<Vec<ChatResponse>, String> {
    println!("Sending chat via gRPC: {}", message);
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Create request
    let request = video_analysis::ChatRequest {
        message,
        session_id: session_id.unwrap_or_else(|| format!("session_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis())),
        video_id,
    };
    
    // Call gRPC Chat method (returns a stream)
    match client.chat(request).await {
        Ok(response) => {
            let mut stream = response.into_inner();
            let mut responses = Vec::new();
            
            // Collect all streamed responses
            while let Ok(Some(msg)) = stream.message().await {
                responses.push(ChatResponse {
                    text: msg.message,
                    sender: msg.sender,
                    timestamp: msg.timestamp as u64,
                    agent_used: msg.agent_used.map(|agent| {
                        serde_json::json!({
                            "agent_name": agent.agent_name,
                            "action": agent.action,
                            "metadata": agent.metadata
                        })
                    }),
                });
            }
            
            println!("Received {} chat responses via gRPC", responses.len());
            Ok(responses)
        }
        Err(e) => {
            Err(format!("gRPC chat failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

/// Tauri command to upload video to backend via gRPC
#[tauri::command]
async fn upload_video(
    filename: String,
    data: Vec<u8>,
    session_id: Option<String>,
) -> Result<UploadResponse, String> {
    println!("Uploading video via gRPC: {} ({} bytes)", filename, data.len());
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Split data into chunks (1MB each)
    const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks
    let total_chunks = (data.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    println!("Splitting into {} chunks", total_chunks);
    
    // Create vector of chunks
    let mut chunks_vec = Vec::new();
    for (i, chunk) in data.chunks(CHUNK_SIZE).enumerate() {
        chunks_vec.push(VideoChunk {
            data: chunk.to_vec(),
            filename: filename.clone(),
            chunk_index: i as i64,
            is_final: i == total_chunks - 1,
            session_id: session_id.clone(),
        });
    }
    
    // Convert to tokio stream
    let stream = tokio_stream::iter(chunks_vec);
    
    // Upload via gRPC
    match client.upload_video(stream).await {
        Ok(response) => {
            let upload_response = response.into_inner();
            println!("Upload successful: {}", upload_response.video_id);
            
            // Convert response
            let metadata = upload_response.metadata.map(|m| VideoMetadata {
                filename: m.filename,
                duration_ms: m.duration_ms,
                width: m.width,
                height: m.height,
                fps: m.fps,
                file_size: m.file_size,
            });
            
            Ok(UploadResponse {
                success: upload_response.success,
                video_id: upload_response.video_id,
                message: upload_response.message,
                metadata,
            })
        }
        Err(e) => {
            Err(format!("gRPC upload failed: {}", e))
        }
    }
}

/// Tauri command to generate report via gRPC
#[tauri::command]
async fn generate_report(
    video_id: String,
    format: String,
    sections: Option<Vec<String>>,
) -> Result<ReportResponse, String> {
    println!("Generating {} report for video: {}", format, video_id);
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Map format string to protobuf enum
    let report_format = match format.to_lowercase().as_str() {
        "pdf" => 0,  // PDF = 0 in enum
        "pptx" => 1, // PPTX = 1 in enum
        _ => return Err(format!("Invalid format: {}. Use 'pdf' or 'pptx'", format)),
    };
    
    // Create request
    let request = video_analysis::ReportRequest {
        video_id: video_id.clone(),
        format: report_format,
        template: None,
        sections: sections.unwrap_or_default(),
    };
    
    // Call gRPC GenerateReport method
    match client.generate_report(request).await {
        Ok(response) => {
            let report_response = response.into_inner();
            println!("Report generated: {}", report_response.filename);
            
            Ok(ReportResponse {
                success: report_response.success,
                message: report_response.message,
                report_data: report_response.report_data,
                filename: report_response.filename,
            })
        }
        Err(e) => {
            Err(format!("gRPC generate report failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

/// Tauri command to get analysis status via gRPC
#[tauri::command]
async fn get_analysis_status(
    video_id: String,
) -> Result<AnalysisStatusResponse, String> {
    println!("Getting analysis status for video: {}", video_id);
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Create request
    let request = video_analysis::StatusRequest {
        video_id: video_id.clone(),
    };
    
    // Call gRPC GetAnalysisStatus method
    match client.get_analysis_status(request).await {
        Ok(response) => {
            let status = response.into_inner();
            
            // Map stage enum to string
            let stage_name = match status.stage {
                0 => "UPLOADED",
                1 => "EXTRACTING_AUDIO",
                2 => "TRANSCRIBING",
                3 => "ANALYZING_FRAMES",
                4 => "EXTRACTING_TEXT",
                5 => "GENERATING_SUMMARY",
                6 => "COMPLETED",
                7 => "FAILED",
                _ => "UNKNOWN",
            };
            
            println!("Status: {} - {}", stage_name, status.message);
            
            Ok(AnalysisStatusResponse {
                video_id: status.video_id,
                stage: stage_name.to_string(),
                progress: status.progress,
                message: status.message,
            })
        }
        Err(e) => {
            Err(format!("gRPC get status failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

/// Tauri command to create a new session via gRPC
#[tauri::command]
async fn create_session(
    title: String,
) -> Result<SessionResponse, String> {
    println!("Creating new session: {}", title);
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Create request
    let request = video_analysis::CreateSessionRequest {
        title: Some(title.clone()),
    };
    
    // Call gRPC CreateSession method
    match client.create_session(request).await {
        Ok(response) => {
            let session = response.into_inner();
            println!("Session created: {}", session.session_id);
            
            Ok(SessionResponse {
                session_id: session.session_id,
                title: session.title,
                created_at: session.created_at,
                updated_at: session.updated_at,
                video_id: session.video_id,
                video_filename: session.video_filename,
                video_metadata: None, // New sessions don't have video yet
            })
        }
        Err(e) => {
            Err(format!("gRPC create session failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

/// Tauri command to list all sessions via gRPC
#[tauri::command]
async fn list_sessions() -> Result<SessionListResponse, String> {
    println!("Listing all sessions");
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Create request
    let request = video_analysis::ListSessionsRequest {};
    
    // Call gRPC ListSessions method
    match client.list_sessions(request).await {
        Ok(response) => {
            let session_list = response.into_inner();
            println!("Found {} sessions", session_list.sessions.len());
            
            let sessions: Vec<SessionResponse> = session_list.sessions.into_iter().map(|s| {
                // Extract video_metadata if available
                let video_metadata = s.video_metadata.map(|vm| VideoMetadata {
                    filename: vm.filename,
                    duration_ms: vm.duration_ms,
                    width: vm.width,
                    height: vm.height,
                    fps: vm.fps,
                    file_size: vm.file_size,
                });
                
                SessionResponse {
                    session_id: s.session_id,
                    title: s.title,
                    created_at: s.created_at,
                    updated_at: s.updated_at,
                    video_id: s.video_id,
                    video_filename: s.video_filename,
                    video_metadata,
                }
            }).collect();
            
            Ok(SessionListResponse { sessions })
        }
        Err(e) => {
            Err(format!("gRPC list sessions failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

/// Tauri command to load a session with history via gRPC
#[tauri::command]
async fn load_session(
    session_id: String,
) -> Result<SessionDataResponse, String> {
    println!("Loading session: {}", session_id);
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Create request
    let request = video_analysis::LoadSessionRequest {
        session_id: session_id.clone(),
    };
    
    // Call gRPC LoadSession method
    match client.load_session(request).await {
        Ok(response) => {
            let session_data = response.into_inner();
            println!("Session loaded with {} messages", session_data.messages.len());
            
            let session = session_data.session.ok_or("No session data returned")?;
            
            let messages: Vec<ChatMessageData> = session_data.messages.into_iter().map(|m| {
                ChatMessageData {
                    role: m.role,
                    content: m.content,
                    timestamp: m.timestamp,
                }
            }).collect();
            
            // Extract video_metadata if available
            let video_metadata = session.video_metadata.map(|vm| VideoMetadata {
                filename: vm.filename,
                duration_ms: vm.duration_ms,
                width: vm.width,
                height: vm.height,
                fps: vm.fps,
                file_size: vm.file_size,
            });
            
            Ok(SessionDataResponse {
                session: SessionResponse {
                    session_id: session.session_id,
                    title: session.title,
                    created_at: session.created_at,
                    updated_at: session.updated_at,
                    video_id: session.video_id,
                    video_filename: session.video_filename,
                    video_metadata,
                },
                messages,
                has_cache: session_data.has_cache,
                cache_summary: session_data.cache_summary,
            })
        }
        Err(e) => {
            Err(format!("gRPC load session failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

/// Tauri command to delete a session via gRPC
#[tauri::command]
async fn delete_session(
    session_id: String,
) -> Result<DeleteSessionResponse, String> {
    println!("Deleting session: {}", session_id);
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Create request
    let request = video_analysis::DeleteSessionRequest {
        session_id: session_id.clone(),
    };
    
    // Call gRPC DeleteSession method
    match client.delete_session(request).await {
        Ok(response) => {
            let delete_result = response.into_inner();
            println!("Delete result: {}", delete_result.message);
            
            Ok(DeleteSessionResponse {
                success: delete_result.success,
                message: delete_result.message,
            })
        }
        Err(e) => {
            Err(format!("gRPC delete session failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

/// Tauri command to update a session via gRPC
#[tauri::command]
async fn update_session(
    session_id: String,
    title: String,
) -> Result<UpdateSessionResponse, String> {
    println!("Updating session: {} with title: {}", session_id, title);
    
    // Create gRPC client
    let mut client = create_grpc_client().await?;
    
    // Create request
    let request = video_analysis::UpdateSessionRequest {
        session_id: session_id.clone(),
        title: title.clone(),
    };
    
    // Call gRPC UpdateSession method
    match client.update_session(request).await {
        Ok(response) => {
            let update_result = response.into_inner();
            println!("Update result: {}", update_result.message);
            
            Ok(UpdateSessionResponse {
                success: update_result.success,
                message: update_result.message,
            })
        }
        Err(e) => {
            Err(format!("gRPC update session failed: {}. Is the backend running on port 50051?", e))
        }
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            grpc_chat, 
            upload_video, 
            generate_report, 
            get_analysis_status,
            create_session,
            list_sessions,
            load_session,
            delete_session,
            update_session
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}


