"""Pipeline script for parsing PDF documents."""

from pathlib import Path
from parser import create_parser
from core.logger import logger
from core.config import config



def main():
    """Main parsing pipeline function."""
    log = logger.get_logger(__name__)
    
    try:
        # Create parser using configuration from .env file
        log.info(f"Input storage: {config.INPUT_STORAGE}, Output storage: {config.OUTPUT_STORAGE}, OCR engine: {config.OCR_ENGINE}")
        parser = create_parser()  # Now automatically uses config values
        
        # Get input and output paths from config (these are the defaults)
        input_path = config.INPUT_PATH
        output_path = config.OUTPUT_PATH
        
        log.info(f"Input path: {input_path}")
        log.info(f"Output path: {output_path}")
        log.info(f"Parallel processing: {'ON' if config.PARALLEL_PROCESSING else 'OFF'}")
        if config.PARALLEL_PROCESSING:
            log.info(f"Max workers: {config.MAX_WORKERS}")
        
        # Use batch_parse function which now uses config paths by default
        result = parser.batch_parse()
        
        if result['status'] == 'error':
            log.error(f"❌ Batch processing failed: {result['error']}")
            return 1
        
        # Display results summary
        total_files = result['processed']
        successful = result['successful']
        failed = result['failed']
        skipped = result.get('skipped', 0)
        processing_time = result.get('processing_time', 0)
        batch_metrics = result.get('batch_metrics', {})
        
        log.info(f"🏁 Processing Summary:")
        log.info(f"   📁 Total files: {total_files}")
        log.info(f"   ✅ Successfully processed: {successful}")
        log.info(f"   ⏭️  Skipped (already processed): {skipped}")
        log.info(f"   ❌ Failed: {failed}")
        log.info(f"   ⏱️  Total time: {processing_time}s")
        
        # Show performance metrics if available
        if batch_metrics:
            efficiency = batch_metrics.get('efficiency_percent', 0)
            avg_time = batch_metrics.get('average_file_time', 0)
            parallel_enabled = result.get('parallel_enabled', False)
            workers_used = result.get('workers_used', 1)
            
            log.info(f"   🚀 Parallel processing: {'ON' if parallel_enabled else 'OFF'}")
            if parallel_enabled:
                log.info(f"   👥 Workers used: {workers_used}")
                log.info(f"   ⚡ Efficiency: {efficiency}%")
            log.info(f"   📊 Average per file: {avg_time}s")
        
        if successful > 0:
            log.info(f"✅ {successful} files processed successfully")
        
        if failed > 0:
            log.warning(f"❌ {failed} files failed to process")
            # Show details of failed files
            for file_result in result.get('results', []):
                if file_result.get('status') == 'error':
                    log.error(f"   Failed: {file_result.get('source_file', 'Unknown')} - {file_result.get('error', 'Unknown error')}")
        
        # Show successful processing details
        for file_result in result.get('results', []):
            if file_result.get('status') == 'success':
                metadata = file_result.get('metadata', {})
                file_name = Path(file_result.get('source_file', '')).name
                
                log.info(f"✅ {file_name}:")
                log.info(f"   Pages: {metadata.get('page_count', 'Unknown')}")
                log.info(f"   OCR used: {metadata.get('ocr_enabled', False)}")
                
                if metadata.get('images_saved', 0) > 0:
                    log.info(f"   Images extracted: {metadata.get('images_saved', 0)}")
                
                doc_structure = metadata.get('document_structure', {})
                if doc_structure.get('has_tables'):
                    log.info(f"   Tables detected: Yes")
                if doc_structure.get('has_figures'):
                    log.info(f"   Figures detected: Yes")
                if metadata.get('image_descriptions'):
                    log.info(f"   Image descriptions: {len(metadata.get('image_descriptions', []))} generated")
        
        # Return appropriate exit code
        if failed > 0:
            return 1 if successful == 0 else 2  # 1 for total failure, 2 for partial failure
        else:
            return 0
            
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())