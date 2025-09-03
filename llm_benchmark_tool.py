"""
Modern LLM Benchmarking Tool with Streamlit GUI
"""

import json
import os
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import ollama
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Session state'te client nesnesini saklamak için
if "client" not in st.session_state:
    st.session_state.client = ollama.Client(host= DEFAULT_OLLAMA_URL)

# Logları saklamak için bir liste
if "logs" not in st.session_state:
    st.session_state.logs:deque = deque(maxlen= 10)

def clear_logs():
    st.session_state.logs.clear()
    st.session_state.log_placeholder.code("",language="log")

def add_log(msg):
    print(msg)
    st.session_state.logs.append(msg)
    logs:list = list(st.session_state.logs)
    logs.reverse()
    log_text = "\n".join(logs)
    st.session_state.log_placeholder.code(log_text,language="log")

@dataclass
class GPUInfo:
    """Data class for GPU information."""
    name: str
    total_memory: int  # in MB


@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""
    gpu_name: str
    gpu_memory: int
    model: str
    mean_output_speed: float
    mean_prompt_speed: float


class SystemInfo:
    """Handle system information detection and GPU queries."""
    
    @staticmethod
    def is_jetson_device() -> bool:
        """Check if running on NVIDIA Jetson device."""
        return Path("/etc/nv_tegra_release").exists()
    
    @staticmethod
    def get_gpu_info() -> Optional[GPUInfo]:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            if lines:
                output = lines[0].split(',')
                return GPUInfo(
                    name=output[0].strip(),
                    total_memory=int(output[1].strip())
                )
        except (subprocess.CalledProcessError, IndexError, ValueError) as e:
            st.error(f"Error getting GPU info: {e}")
        
        return None
    
    @staticmethod
    def jetson_get_gpu_info() -> Optional[GPUInfo]:
        """Get GPU information for Jetson devices."""
        try:
            from jtop import jtop
            
            with jtop() as jetson:
                gpu_name = jetson.board['hardware']['Model']
                total_memory = jetson.memory['RAM']['tot'] // 1024
                
            return GPUInfo(
                name=gpu_name,
                total_memory=total_memory
            )
        except ImportError:
            st.error("jtop not installed. Install with: pip install jetson-stats")
        except Exception as e:
            st.error(f"Error getting Jetson GPU info: {e}")
        
        return None


class ModelManager:
    """Handle model operations and benchmarking."""

    @staticmethod
    def get_existing_models() -> List[str]:
        """Get list of existing models with normalized names."""
        try:           

            models = [model.model for model in st.session_state.client.list()['models']]
            print(models)
            # Also create versions without :latest tag for matching
            normalized_models = []
            for model in models:
                normalized_models.append(model)
                # Add version without :latest tag
                if model.endswith(':latest'):
                    normalized_models.append(model.replace(':latest', ''))
                # Add version with :latest tag if not present
                elif ':' not in model:
                    normalized_models.append(f"{model}:latest")
            
            return list(set(normalized_models))  # Remove duplicates
        except Exception as e:
            st.error(f"Error getting existing models: {e}")
            return []
    
    @staticmethod
    def normalize_model_name(model_name: str, existing_models: List[str]) -> Optional[str]:
        """Find the correct model name from existing models."""
        # Direct match
        if model_name in existing_models:
            return model_name
        
        # Try with :latest tag
        if f"{model_name}:latest" in existing_models:
            return f"{model_name}:latest"
        
        # Try without :latest tag
        if model_name.endswith(':latest'):
            base_name = model_name.replace(':latest', '')
            if base_name in existing_models:
                return base_name
        
        # Case insensitive search
        for existing_model in existing_models:
            if existing_model.lower() == model_name.lower():
                return existing_model
            if existing_model.lower().replace(':latest', '') == model_name.lower():
                return existing_model
        
        return None    
    
    @staticmethod
    def load_models_from_json(file_path: str) -> Dict[str, float]:
        """Load model configurations from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Model configuration file '{file_path}' not found.")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON file: {e}")
            return {}
    
    @staticmethod
    def unload_model(model: str) -> bool:
        """Unload model from memory."""
        try:
            url = 'http://localhost:11434/api/generate'
            headers = {'Content-Type': 'application/json'}
            data = json.dumps({"model": model, "keep_alive": 0})
            
            response = requests.post(url, headers=headers, data=data, timeout=30)
            return response.status_code == 200
        except requests.RequestException as e:
            st.warning(f"Error unloading model {model}: {e}")
            return False
      
    @staticmethod
    def pull_model(model: str, progress_bar, status_text) -> bool:
        """Pull model with progress tracking."""
        try:
            status_text.text(f"Pulling model: {model}")
            pull_resp = st.session_state.client.pull(model, stream=True)
            
            for chunk in pull_resp:
                if 'completed' in chunk and 'total' in chunk:
                    progress = chunk['completed'] / chunk['total']
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading {model}: {progress:.1%}")
            
            progress_bar.progress(1.0)
            status_text.text(f"Successfully pulled {model}")
            return True
            
        except Exception as e:
            st.error(f"Error pulling model {model}: {e}")
            return False


class LLMBenchmark:
    """Main benchmarking class."""
    
    def __init__(self):
        self.system_info = SystemInfo()
        self.model_manager = ModelManager()
    
    @staticmethod
    def send_request(model_name: str, prompt: str) -> Optional[ollama.ChatResponse]:
        """Send request to model and return response."""
        try:
            # st.write(f"    🔄 Sending request to {model_name}...")
            response = st.session_state.client.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }]
            )
            # st.write(f"    ✅ Response received from {model_name}")
            return response
        except Exception as e:
            add_log(f"❌ Error sending request to {model_name}: {e}")
            return None
    
    @staticmethod
    def calculate_speed(response_data: Dict) -> Tuple[float, float]:
        """Calculate generation and prompt evaluation speeds."""
        # Generation speed
        eval_count = response_data.get("eval_count", 0)
        eval_duration = response_data.get("eval_duration", 1)
        eval_duration_seconds = eval_duration / 1e9
        speed_eval = eval_count / eval_duration_seconds if eval_duration_seconds > 0 else 0
        
        # Prompt evaluation speed
        prompt_eval_count = response_data.get("prompt_eval_count", 0)
        prompt_eval_duration = response_data.get("prompt_eval_duration", 1)
        prompt_eval_duration_seconds = prompt_eval_duration / 1e9
        speed_prompt_eval = prompt_eval_count / prompt_eval_duration_seconds if prompt_eval_duration_seconds > 0 else 0
        
        return speed_eval, speed_prompt_eval

    def run_benchmark(
        self,
        models: Dict[str, float],
        prompts: List[str],
        gpu_info: GPUInfo,
        progress_callback=None
    ) -> List[BenchmarkResult]:
        """Run benchmark on selected models."""
        results = []
        existing_models = self.model_manager.get_existing_models()
        
        add_log(f"🔍 Starting benchmark for {len(models)} models...")
        print("**Selected models:**", list(models.keys()))
        print("**Existing models:**", existing_models)
        
        for i, (model, memory_requirement) in enumerate(models.items()):
            if progress_callback:
                progress_callback(i, len(models), f"Processing {model}")
            
            add_log(f"📊 **Processing Model {i+1}/{len(models)}: {model}**")
            
            # Check memory requirement
            available_memory_gb = gpu_info.total_memory / 1024
            if available_memory_gb <= memory_requirement:
                error_msg = f"❌ Insufficient memory for {model}. Required: {memory_requirement}GB, Available: {available_memory_gb:.1f}GB"
                add_log(error_msg)
                continue
            else:
                add_log(f"✅ Memory check passed: {available_memory_gb:.1f}GB available, {memory_requirement}GB required")
            
            # Check if model exists (with smart name matching)
            actual_model_name = ModelManager.normalize_model_name(model, existing_models)
            if not actual_model_name:
                add_log(f"⚠️ Model {model} not found in existing models.")
                add_log("**Available models:**")
                for existing_model in existing_models:
                    print(f"  - {existing_model}")
                print("💡 **Tip:** Pull the model first using `ollama pull <model_name>`")
                continue
            else:
                if actual_model_name != model:
                    add_log(f"🔄 Using model name: `{actual_model_name}` (instead of `{model}`)")
                add_log(f"✅ Model {actual_model_name} found and ready")
            
            # Run benchmark with the actual model name
            add_log(f"🚀 Running benchmark with {len(prompts)} prompts...")
            total_output_speed = 0
            total_prompt_speed = 0
            successful_prompts = 0
            
            for j, prompt in enumerate(prompts):
                try:
                    # print(f"  📝 Prompt {j+1}/{len(prompts)}: {prompt[:50]}...")
                    response_data = self.send_request(actual_model_name, prompt)  # Use actual name
                    if not response_data:
                        add_log(f"❌ Failed to get response for prompt {j+1}")
                        continue
                    
                    output_speed, prompt_speed = self.calculate_speed(response_data)
                    total_output_speed += output_speed
                    total_prompt_speed += prompt_speed
                    successful_prompts += 1
                    add_log(f"✅ Prompt {j+1} completed - Output: {output_speed:.1f} tok/sec, Prompt: {prompt_speed:.1f} tok/sec")                    
                    
                except Exception as e:
                    add_log(f"❌ Error processing prompt {j+1}: {str(e)}")
                    continue
            
            if successful_prompts > 0:
                mean_output_speed = total_output_speed / successful_prompts
                mean_prompt_speed = total_prompt_speed / successful_prompts
                
                result = BenchmarkResult(
                    gpu_name=gpu_info.name,
                    gpu_memory=gpu_info.total_memory,
                    model=model,
                    mean_output_speed=round(mean_output_speed, 3),
                    mean_prompt_speed=round(mean_prompt_speed, 3)
                )
                results.append(result)
                
                add_log(f"🎯 **{model} completed successfully!**")
                add_log(f"📊 Mean Output Speed: {mean_output_speed:.1f} tokens/sec")
                add_log(f"⚡ Mean Prompt Speed: {mean_prompt_speed:.1f} tokens/sec")
                add_log(f"✅ Successful prompts: {successful_prompts}/{len(prompts)}")
                
            else:
                add_log(f"❌ No successful prompts for {model}")
            
            # Unload model
            add_log(f"🔄 Unloading {actual_model_name}...")
            if self.model_manager.unload_model(actual_model_name):
                add_log(f"✅ {actual_model_name} unloaded successfully")
            else:
                add_log(f"⚠️ Failed to unload {actual_model_name}")
            
            print("---")  # Separator between models
        
        add_log(f"🏁 Benchmark completed! Results for {len(results)}/{len(models)} models")
        return results

def create_benchmark_charts(results_df: pd.DataFrame):
    """Create interactive Plotly charts for benchmark results."""
    
    if results_df.empty:
        st.warning("No data available for charts.")
        return
    
    # Sort by output speed for better visualization
    results_df_sorted = results_df.sort_values('Mean Output Speed (tokens/sec)', ascending=True)
    
    # Color palette for consistent styling
    colors = px.colors.qualitative.Set3
    
    # Chart 1: Mean Output Speed
    st.subheader("📊 Mean Output Speed Comparison")
    
    fig_output = px.bar(
        results_df_sorted,
        x='Mean Output Speed (tokens/sec)',
        y='Model',
        orientation='h',
        title='Model Performance - Token Generation Speed',
        labels={
            'Mean Output Speed (tokens/sec)': 'Tokens per Second',
            'Model': 'Model Name'
        },
        color='Mean Output Speed (tokens/sec)',
        color_continuous_scale='Blues',
        text='Mean Output Speed (tokens/sec)'
    )
    
    fig_output.update_traces(
        texttemplate='%{text:.1f}',
        textposition='inside',
        textfont_size=12,
        textfont_color='white'
    )
    
    fig_output.update_layout(
        height=max(400, len(results_df) * 50),
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=16,
        title_x=0.5
    )
    
    fig_output.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig_output.update_yaxes(
        showgrid=False
    )
    
    st.plotly_chart(fig_output, width='stretch')
    
    # Chart 2: Mean Prompt Speed
    st.subheader("⚡ Mean Prompt Processing Speed Comparison")
    
    results_df_prompt_sorted = results_df.sort_values('Mean Prompt Speed (tokens/sec)', ascending=True)
    
    fig_prompt = px.bar(
        results_df_prompt_sorted,
        x='Mean Prompt Speed (tokens/sec)',
        y='Model',
        orientation='h',
        title='Model Performance - Prompt Processing Speed',
        labels={
            'Mean Prompt Speed (tokens/sec)': 'Tokens per Second',
            'Model': 'Model Name'
        },
        color='Mean Prompt Speed (tokens/sec)',
        color_continuous_scale='Blues',
        text='Mean Prompt Speed (tokens/sec)'
    )
    
    fig_prompt.update_traces(
        texttemplate='%{text:.1f}',
        textposition='inside',
        textfont_size=12,
        textfont_color='white'
    )
    
    fig_prompt.update_layout(
        height=max(400, len(results_df) * 50),
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=16,
        title_x=0.5
    )
    
    fig_prompt.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig_prompt.update_yaxes(
        showgrid=False
    )
    
    st.plotly_chart(fig_prompt, width='stretch')
    
    # Chart 3: Scatter Plot - Output vs Prompt Speed
    st.subheader("🎯 Performance Correlation")
    
    fig_scatter = px.scatter(
        results_df,
        x='Mean Prompt Speed (tokens/sec)',
        y='Mean Output Speed (tokens/sec)',
        size='Mean Output Speed (tokens/sec)',
        hover_name='Model',
        title='Output Speed vs Prompt Speed Correlation',
        labels={
            'Mean Prompt Speed (tokens/sec)': 'Prompt Processing Speed (tokens/sec)',
            'Mean Output Speed (tokens/sec)': 'Output Generation Speed (tokens/sec)'
        },
        color='Model',
        size_max=20
    )
    
    fig_scatter.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=16,
        title_x=0.5,
        font=dict(size=12)
    )
    
    fig_scatter.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig_scatter.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    st.plotly_chart(fig_scatter, width='stretch')
    
    # Performance Summary Cards
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_output = results_df.loc[results_df['Mean Output Speed (tokens/sec)'].idxmax()]
        st.metric(
            "🚀 Fastest Output",
            best_output['Model'],
            f"{best_output['Mean Output Speed (tokens/sec)']:.1f} tok/sec"
        )
    
    with col2:
        best_prompt = results_df.loc[results_df['Mean Prompt Speed (tokens/sec)'].idxmax()]
        st.metric(
            "⚡ Fastest Prompt",
            best_prompt['Model'],
            f"{best_prompt['Mean Prompt Speed (tokens/sec)']:.1f} tok/sec"
        )
    
    with col3:
        avg_output = results_df['Mean Output Speed (tokens/sec)'].mean()
        st.metric(
            "📊 Avg Output Speed",
            f"{avg_output:.1f}",
            "tokens/sec"
        )
    
    with col4:
        avg_prompt = results_df['Mean Prompt Speed (tokens/sec)'].mean()
        st.metric(
            "📊 Avg Prompt Speed", 
            f"{avg_prompt:.1f}",
            "tokens/sec"
        )


def main():
    """Main Streamlit application."""
    # st.markdown(f'<style> .sidebar {{width: 90%;}} </style>', unsafe_allow_html=True)
    st.set_page_config(
        page_title="LLM Benchmark Tool",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 LLM Benchmark Tool")
    st.markdown("Benchmark LLM models using Ollama with real-time progress tracking")
    
    # Initialize benchmark
    benchmark = LLMBenchmark()
    
    
    st.sidebar.image("images/CORDATUS_LOGO.png")
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # GPU Information
    st.sidebar.subheader("GPU Information")
    gpu_info = None

    if SystemInfo.is_jetson_device():
        gpu_info = SystemInfo.jetson_get_gpu_info()
        st.sidebar.info("🎯 Jetson device detected")
    else:
        gpu_info = SystemInfo.get_gpu_info()
    
    if gpu_info:
        st.sidebar.success(f"**GPU:** {gpu_info.name}")
        st.sidebar.success(f"**Memory:** {gpu_info.total_memory:,} MB ({gpu_info.total_memory/1024:.1f} GB)")
    else:
        st.sidebar.error("⚠️ Could not detect GPU information")
        st.stop()
     
    models = ModelManager.load_models_from_json("models.json")
    if not models:
        st.stop()
    

    prompts_file = "test_prompts.txt"
    try:
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        st.sidebar.error(f"Error reading prompts file: {e}")
        st.stop()

    if "log_placeholder" not in st.session_state:
        with st.container(height= 500):
            st.session_state.log_placeholder = st.sidebar.empty()
        
    # Main content
    st.header("Models compatible with GPU memory (VRAM) requirements")
    
    # Filter models by memory
    available_models = {
        model: memory for model, memory in models.items()
        if gpu_info.total_memory / 1024 > memory
    }
    
    insufficient_models = {
        model: memory for model, memory in models.items()
        if gpu_info.total_memory / 1024 <= memory }

    
    if available_models:
        df_available = pd.DataFrame([
            {"Model": model, "Memory Required (GB)": memory, "Status": "✅"}
            for model, memory in available_models.items()
        ])
        st.dataframe(df_available, width='stretch')
    
    if insufficient_models:
        st.subheader("Insufficient Memory")
        df_insufficient = pd.DataFrame([
            {"Model": model, "Memory Required (GB)": memory, "Status": "❌ Insufficient Memory"}
            for model, memory in insufficient_models.items()
        ])
        st.dataframe(df_insufficient, width='stretch')
    
    # Model selection
    st.header("Select Models to Benchmark")
    selected_models = {}
    
    cols = st.columns(3)
    for i, (model, memory) in enumerate(available_models.items()):
        with cols[i % 3]:
            if st.checkbox(f"{model} ({memory}GB)", key=model):
                selected_models[model] = memory
    
    # Run benchmark
    if selected_models and st.button("🚀 Start Benchmark", type="primary"):
        st.header("Benchmark Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        def progress_callback(current, total, message):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"{message} ({current}/{total})")
        
        # Check for missing models and pull them
        existing_models = ModelManager.get_existing_models()
        missing_models = [model for model in selected_models.keys() if model not in existing_models]
        
        if missing_models:
            st.sidebar.subheader("Pulling Missing Models")
            for model in missing_models:
                model_progress = st.sidebar.progress(0)
                model_status = st.sidebar.empty()
                
                if ModelManager.pull_model(model, model_progress, model_status):
                    time.sleep(1)  # Brief pause between models
        
        # Run benchmark
        clear_logs()
        results = benchmark.run_benchmark(
            selected_models,
            prompts,
            gpu_info,
            progress_callback
        )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Benchmark completed!")
        
        if results:
            st.header("🎯 Benchmark Results")
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Convert to DataFrame
            results_df = pd.DataFrame([
                {
                    "Date": now,
                    "Model": result.model,
                    "Mean Output Speed (tokens/sec)": result.mean_output_speed,
                    "Mean Prompt Speed (tokens/sec)": result.mean_prompt_speed,
                    "GPU": result.gpu_name,
                    "GPU Memory (MB)": result.gpu_memory
                }
                for result in results
            ])
            
            # Show results table
            st.subheader("📋 Detailed Results Table")
            st.dataframe(results_df, width='stretch')
            
            # Create and display charts
            create_benchmark_charts(results_df)
            
            # Save results
            output_file = "benchmark_results.csv"
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            else:
                combined_df = results_df
            
            combined_df.to_csv(output_file, index=False)
            st.success(f"Results saved to {output_file}")
            
            # Download button
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="benchmark_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No results generated. Please check your configuration.")


if __name__ == "__main__":
    main()
