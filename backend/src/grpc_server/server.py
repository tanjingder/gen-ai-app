"""
gRPC Server Implementation
"""
import asyncio
import grpc
from concurrent import futures
from loguru import logger

from .service import VideoAnalysisServicer
from ..utils.config import settings


async def serve():
    """Start the gRPC server"""
    # Generate proto files first (before imports)
    logger.info("Compiling protobuf definitions...")
    compile_proto()
    
    # Import generated modules after compilation
    from . import video_analysis_pb2_grpc
    
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
    )
    
    # Add servicer
    video_analysis_pb2_grpc.add_VideoAnalysisServiceServicer_to_server(
        VideoAnalysisServicer(),
        server
    )
    
    # Start server
    listen_addr = f'{settings.GRPC_HOST}:{settings.GRPC_PORT}'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    
    logger.info("gRPC server is running. Press Ctrl+C to stop.")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(grace=5)


def compile_proto():
    """Compile protobuf files"""
    from grpc_tools import protoc
    import os
    from pathlib import Path
    
    # Get paths
    project_root = Path(__file__).parent.parent.parent.parent
    proto_dir = project_root / "proto"
    output_dir = Path(__file__).parent
    
    proto_file = proto_dir / "video_analysis.proto"
    
    if not proto_file.exists():
        logger.error(f"Proto file not found: {proto_file}")
        raise FileNotFoundError(f"Proto file not found: {proto_file}")
    
    # Compile
    logger.info(f"Compiling {proto_file}")
    
    result = protoc.main([
        'grpc_tools.protoc',
        f'--proto_path={proto_dir}',
        f'--python_out={output_dir}',
        f'--grpc_python_out={output_dir}',
        str(proto_file)
    ])
    
    if result != 0:
        logger.error("Failed to compile proto files")
        raise RuntimeError("Proto compilation failed")
    
    # Fix imports in generated grpc file
    grpc_file = output_dir / "video_analysis_pb2_grpc.py"
    if grpc_file.exists():
        logger.info("Fixing imports in generated gRPC file...")
        content = grpc_file.read_text()
        # Change absolute import to relative import
        content = content.replace(
            "import video_analysis_pb2 as video__analysis__pb2",
            "from . import video_analysis_pb2 as video__analysis__pb2"
        )
        grpc_file.write_text(content)
    
    logger.info("Proto files compiled successfully")
