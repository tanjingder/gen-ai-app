fn main() {
    // Compile protobuf
    tonic_build::configure()
        .build_server(false) // We only need the client
        .build_client(true)
        .compile(
            &["../../proto/video_analysis.proto"],
            &["../../proto"],
        )
        .expect("Failed to compile proto files");
    
    tauri_build::build()
}
