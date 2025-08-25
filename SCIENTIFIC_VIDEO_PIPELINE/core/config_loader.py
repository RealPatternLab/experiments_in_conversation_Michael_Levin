#!/usr/bin/env python3
"""
Configuration Loader for Unified Video Pipeline
Loads pipeline-specific configurations from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PipelineConfigLoader:
    """Loads and manages pipeline configurations"""
    
    def __init__(self, config_dir: str = "core/pipeline_configs"):
        """
        Initialize the configuration loader
        
        Args:
            config_dir: Directory containing pipeline configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all pipeline configurations"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        
        # Find all YAML configuration files
        config_files = list(self.config_dir.glob("*.yaml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                pipeline_type = config.get('pipeline_type')
                if pipeline_type:
                    self.configs[pipeline_type] = config
                    print(f"✅ Loaded configuration for {pipeline_type}")
                else:
                    print(f"⚠️ Configuration file {config_file.name} missing pipeline_type")
                    
            except Exception as e:
                print(f"❌ Failed to load configuration from {config_file.name}: {e}")
    
    def get_config(self, pipeline_type: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific pipeline type
        
        Args:
            pipeline_type: The pipeline type (e.g., 'formal_presentations', 'conversations_1_on_2')
            
        Returns:
            Pipeline configuration dictionary or None if not found
        """
        return self.configs.get(pipeline_type)
    
    def list_available_pipelines(self) -> list:
        """List all available pipeline types"""
        return list(self.configs.keys())
    
    def validate_config(self, pipeline_type: str) -> bool:
        """
        Validate a pipeline configuration
        
        Args:
            pipeline_type: The pipeline type to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        config = self.get_config(pipeline_type)
        if not config:
            return False
        
        # Check required fields
        required_fields = [
            'pipeline_type',
            'pipeline_name',
            'speaker_count',
            'transcription_service'
        ]
        
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate speaker count
        speaker_count = config.get('speaker_count', 0)
        if speaker_count < 1:
            print(f"❌ Invalid speaker count: {speaker_count}")
            return False
        
        # Validate transcription service
        transcription_service = config.get('transcription_service')
        if transcription_service not in ['assemblyai', 'whisper', 'other']:
            print(f"⚠️ Unknown transcription service: {transcription_service}")
        
        print(f"✅ Configuration for {pipeline_type} is valid")
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get a default configuration template"""
        return {
            'pipeline_type': 'custom',
            'pipeline_name': 'Custom Pipeline',
            'description': 'Custom pipeline configuration',
            'speaker_count': 1,
            'transcription_service': 'assemblyai',
            'llm_enhancement': False,
            'chunking_strategy': 'semantic',
            'frame_extraction': {
                'enabled': True,
                'interval_seconds': 15
            }
        }
    
    def create_custom_config(self, pipeline_type: str, **kwargs) -> Dict[str, Any]:
        """
        Create a custom configuration by merging with defaults
        
        Args:
            pipeline_type: The pipeline type
            **kwargs: Configuration overrides
            
        Returns:
            Custom configuration dictionary
        """
        # Start with default config
        config = self.get_default_config()
        
        # Override with custom values
        config.update(kwargs)
        config['pipeline_type'] = pipeline_type
        
        return config
    
    def save_config(self, pipeline_type: str, config: Dict[str, Any], filename: str = None):
        """
        Save a configuration to a YAML file
        
        Args:
            pipeline_type: The pipeline type
            config: The configuration dictionary
            filename: Optional custom filename
        """
        if not filename:
            filename = f"{pipeline_type}.yaml"
        
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            # Reload configs
            self._load_all_configs()
            print(f"✅ Configuration saved to {config_path}")
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    def print_config_summary(self, pipeline_type: str):
        """Print a summary of a pipeline configuration"""
        config = self.get_config(pipeline_type)
        if not config:
            print(f"❌ Configuration not found for {pipeline_type}")
            return
        
        print(f"\n📋 Configuration Summary for {pipeline_type}")
        print(f"  Pipeline Name: {config.get('pipeline_name', 'N/A')}")
        print(f"  Description: {config.get('description', 'N/A')}")
        print(f"  Speaker Count: {config.get('speaker_count', 'N/A')}")
        print(f"  Transcription: {config.get('transcription_service', 'N/A')}")
        print(f"  LLM Enhancement: {config.get('llm_enhancement', False)}")
        print(f"  Chunking Strategy: {config.get('chunking_strategy', 'N/A')}")
        print(f"  Frame Extraction: {'Enabled' if config.get('frame_extraction', {}).get('enabled', False) else 'Disabled'}")
    
    def print_all_configs_summary(self):
        """Print summary of all available configurations"""
        print(f"\n🚀 Available Pipeline Configurations:")
        print(f"  Configuration directory: {self.config_dir}")
        print(f"  Total configurations: {len(self.configs)}")
        
        for pipeline_type in self.configs:
            config = self.configs[pipeline_type]
            print(f"\n  📁 {pipeline_type}:")
            print(f"    Name: {config.get('pipeline_name', 'N/A')}")
            print(f"    Speakers: {config.get('speaker_count', 'N/A')}")
            print(f"    LLM: {'✅' if config.get('llm_enhancement', False) else '❌'}")
            print(f"    Frames: {'✅' if config.get('frame_extraction', {}).get('enabled', False) else '❌'}")


def load_pipeline_config(pipeline_type: str, config_dir: str = "core/pipeline_configs") -> Dict[str, Any]:
    """
    Convenience function to load a pipeline configuration
    
    Args:
        pipeline_type: The pipeline type to load
        config_dir: Directory containing configuration files
        
    Returns:
        Pipeline configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration not found
    """
    loader = PipelineConfigLoader(config_dir)
    config = loader.get_config(pipeline_type)
    
    if not config:
        available = loader.list_available_pipelines()
        raise FileNotFoundError(
            f"Configuration not found for pipeline type: {pipeline_type}\n"
            f"Available types: {', '.join(available)}"
        )
    
    return config


if __name__ == "__main__":
    # Test the configuration loader
    try:
        loader = PipelineConfigLoader()
        loader.print_all_configs_summary()
        
        # Test loading a specific config
        if loader.configs:
            test_pipeline = list(loader.configs.keys())[0]
            print(f"\n🧪 Testing configuration for: {test_pipeline}")
            loader.print_config_summary(test_pipeline)
            loader.validate_config(test_pipeline)
            
    except Exception as e:
        print(f"❌ Configuration loader test failed: {e}")
