"""
Checkpoint and recovery system for optimization progress
"""
import json
import os
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("gemini_optimizer.checkpoint")

@dataclass
class CheckpointData:
    """Data structure for checkpoint information"""
    timestamp: float
    iteration: int
    current_prompt: str
    best_score: float
    best_prompt: str
    optimization_history: List[Dict[str, Any]]
    error_count: int
    last_error: Optional[str]
    strategy_state: Dict[str, Any]
    progress_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Create from dictionary"""
        return cls(**data)

class CheckpointManager:
    """Manages checkpoints and recovery for optimization process"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", auto_save_interval: int = 300):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_save_interval = auto_save_interval  # seconds
        self.last_save_time = 0
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Current checkpoint data
        self.current_checkpoint: Optional[CheckpointData] = None
        
        logger.info(f"CheckpointManager initialized with dir: {self.checkpoint_dir}")
    
    def save_checkpoint(self, checkpoint_data: CheckpointData, force: bool = False) -> bool:
        """Save checkpoint to disk"""
        try:
            current_time = time.time()
            
            # Check if we should save (auto-save interval or forced)
            if not force and (current_time - self.last_save_time) < self.auto_save_interval:
                return False
            
            # Generate checkpoint filename with timestamp (including microseconds for uniqueness)
            dt = datetime.fromtimestamp(checkpoint_data.timestamp)
            timestamp_str = dt.strftime("%Y%m%d_%H%M%S") + f"_{dt.microsecond:06d}"
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp_str}.json"
            
            # Save checkpoint data
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Also save as latest checkpoint
            latest_file = self.checkpoint_dir / "latest_checkpoint.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.current_checkpoint = checkpoint_data
            self.last_save_time = current_time
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_latest_checkpoint(self) -> Optional[CheckpointData]:
        """Load the latest checkpoint"""
        try:
            latest_file = self.checkpoint_dir / "latest_checkpoint.json"
            
            if not latest_file.exists():
                logger.info("No checkpoint found")
                return None
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = CheckpointData.from_dict(data)
            self.current_checkpoint = checkpoint
            
            logger.info(f"Loaded checkpoint from {datetime.fromtimestamp(checkpoint.timestamp)}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def load_checkpoint_by_timestamp(self, timestamp: str) -> Optional[CheckpointData]:
        """Load checkpoint by timestamp string (YYYYMMDD_HHMMSS)"""
        try:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
            
            if not checkpoint_file.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_file}")
                return None
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = CheckpointData.from_dict(data)
            logger.info(f"Loaded checkpoint: {checkpoint_file}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {timestamp}: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    checkpoints.append({
                        "filename": checkpoint_file.name,
                        "timestamp": data.get("timestamp", 0),
                        "iteration": data.get("iteration", 0),
                        "best_score": data.get("best_score", 0.0),
                        "progress": data.get("progress_percentage", 0.0)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
            
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_count: int = 10):
        """Clean up old checkpoints, keeping only the most recent ones"""
        try:
            checkpoints = self.list_checkpoints()
            
            if len(checkpoints) <= keep_count:
                return
            
            # Remove old checkpoints
            for checkpoint in checkpoints[keep_count:]:
                checkpoint_file = self.checkpoint_dir / checkpoint["filename"]
                try:
                    checkpoint_file.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")
            
            logger.info(f"Cleaned up {len(checkpoints) - keep_count} old checkpoints")
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
    
    def create_checkpoint(self, iteration: int, current_prompt: str, best_score: float, 
                         best_prompt: str, optimization_history: List[Dict[str, Any]], 
                         error_count: int = 0, last_error: Optional[str] = None,
                         strategy_state: Optional[Dict[str, Any]] = None,
                         progress_percentage: float = 0.0) -> CheckpointData:
        """Create a new checkpoint data object"""
        return CheckpointData(
            timestamp=time.time(),
            iteration=iteration,
            current_prompt=current_prompt,
            best_score=best_score,
            best_prompt=best_prompt,
            optimization_history=optimization_history,
            error_count=error_count,
            last_error=last_error,
            strategy_state=strategy_state or {},
            progress_percentage=progress_percentage
        )
    
    def create_incremental_checkpoint(self, base_checkpoint: Optional[CheckpointData] = None, 
                                    **updates) -> CheckpointData:
        """Create incremental checkpoint based on previous state"""
        if base_checkpoint is None:
            base_checkpoint = self.load_latest_checkpoint()
        
        if base_checkpoint is None:
            # Create new checkpoint if no base exists
            return self.create_checkpoint(
                iteration=updates.get('iteration', 0),
                current_prompt=updates.get('current_prompt', ''),
                best_score=updates.get('best_score', 0.0),
                best_prompt=updates.get('best_prompt', ''),
                optimization_history=updates.get('optimization_history', []),
                error_count=updates.get('error_count', 0),
                last_error=updates.get('last_error'),
                strategy_state=updates.get('strategy_state', {}),
                progress_percentage=updates.get('progress_percentage', 0.0)
            )
        
        # Update only specified fields
        return CheckpointData(
            timestamp=time.time(),
            iteration=updates.get('iteration', base_checkpoint.iteration),
            current_prompt=updates.get('current_prompt', base_checkpoint.current_prompt),
            best_score=updates.get('best_score', base_checkpoint.best_score),
            best_prompt=updates.get('best_prompt', base_checkpoint.best_prompt),
            optimization_history=updates.get('optimization_history', base_checkpoint.optimization_history),
            error_count=updates.get('error_count', base_checkpoint.error_count),
            last_error=updates.get('last_error', base_checkpoint.last_error),
            strategy_state=updates.get('strategy_state', base_checkpoint.strategy_state),
            progress_percentage=updates.get('progress_percentage', base_checkpoint.progress_percentage)
        )
    
    def save_progress_checkpoint(self, operation_name: str, progress_data: Dict[str, Any]) -> bool:
        """Save a progress-specific checkpoint"""
        try:
            # Create progress checkpoint with current timestamp
            progress_checkpoint = {
                'timestamp': time.time(),
                'operation': operation_name,
                'progress_data': progress_data,
                'checkpoint_type': 'progress'
            }
            
            # Save to progress-specific file
            progress_file = self.checkpoint_dir / f"progress_{operation_name}.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_checkpoint, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Progress checkpoint saved for {operation_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save progress checkpoint: {e}")
            return False
    
    def load_progress_checkpoint(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Load progress-specific checkpoint"""
        try:
            progress_file = self.checkpoint_dir / f"progress_{operation_name}.json"
            
            if not progress_file.exists():
                return None
            
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_checkpoint = json.load(f)
            
            logger.debug(f"Progress checkpoint loaded for {operation_name}")
            return progress_checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load progress checkpoint: {e}")
            return None
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint statistics"""
        try:
            checkpoints = self.list_checkpoints()
            
            if not checkpoints:
                return {
                    'total_checkpoints': 0,
                    'latest_checkpoint': None,
                    'best_score_checkpoint': None,
                    'checkpoint_frequency': 0,
                    'storage_usage': 0
                }
            
            # Find best score checkpoint
            best_checkpoint = max(checkpoints, key=lambda x: x.get('best_score', 0))
            
            # Calculate checkpoint frequency (checkpoints per hour)
            if len(checkpoints) > 1:
                time_span = checkpoints[0]['timestamp'] - checkpoints[-1]['timestamp']
                frequency = len(checkpoints) / max(time_span / 3600, 1)  # per hour
            else:
                frequency = 0
            
            # Calculate storage usage
            storage_usage = 0
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                storage_usage += checkpoint_file.stat().st_size
            
            return {
                'total_checkpoints': len(checkpoints),
                'latest_checkpoint': checkpoints[0] if checkpoints else None,
                'best_score_checkpoint': best_checkpoint,
                'checkpoint_frequency': frequency,
                'storage_usage': storage_usage,
                'average_score': sum(c.get('best_score', 0) for c in checkpoints) / len(checkpoints),
                'score_trend': self._calculate_score_trend(checkpoints)
            }
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint statistics: {e}")
            return {}
    
    def _calculate_score_trend(self, checkpoints: List[Dict[str, Any]]) -> str:
        """Calculate score trend from checkpoints"""
        if len(checkpoints) < 3:
            return "insufficient_data"
        
        # Get recent scores (last 5 checkpoints)
        recent_scores = [c.get('best_score', 0) for c in checkpoints[:5]]
        
        # Calculate trend
        if len(recent_scores) >= 3:
            early_avg = sum(recent_scores[-3:]) / 3
            late_avg = sum(recent_scores[:3]) / 3
            
            if late_avg > early_avg + 0.05:
                return "improving"
            elif late_avg < early_avg - 0.05:
                return "declining"
            else:
                return "stable"
        
        return "unknown"
    
    def should_auto_save(self) -> bool:
        """Check if auto-save should be triggered"""
        return (time.time() - self.last_save_time) >= self.auto_save_interval
    
    def get_recovery_info(self) -> Optional[Dict[str, Any]]:
        """Get recovery information from latest checkpoint"""
        if not self.current_checkpoint:
            checkpoint = self.load_latest_checkpoint()
            if not checkpoint:
                return None
        else:
            checkpoint = self.current_checkpoint
        
        return {
            "can_recover": True,
            "last_iteration": checkpoint.iteration,
            "best_score_achieved": checkpoint.best_score,
            "progress_percentage": checkpoint.progress_percentage,
            "error_count": checkpoint.error_count,
            "last_error": checkpoint.last_error,
            "checkpoint_age_minutes": (time.time() - checkpoint.timestamp) / 60
        }
    
    def export_checkpoint(self, checkpoint_data: CheckpointData, export_path: str) -> bool:
        """Export checkpoint to a specific path"""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Checkpoint exported to: {export_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export checkpoint: {e}")
            return False
    
    def import_checkpoint(self, import_path: str) -> Optional[CheckpointData]:
        """Import checkpoint from a specific path"""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                logger.error(f"Import file not found: {import_file}")
                return None
            
            with open(import_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint = CheckpointData.from_dict(data)
            logger.info(f"Checkpoint imported from: {import_file}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to import checkpoint: {e}")
            return None

class RecoveryManager:
    """Manages system recovery from failures"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        logger.info("RecoveryManager initialized")
    
    def attempt_recovery(self) -> Optional[Dict[str, Any]]:
        """Attempt to recover from the latest checkpoint"""
        try:
            checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            
            if not checkpoint:
                logger.info("No checkpoint available for recovery")
                return None
            
            # Validate checkpoint data
            if not self._validate_checkpoint(checkpoint):
                logger.error("Checkpoint validation failed")
                return None
            
            recovery_info = {
                "checkpoint": checkpoint,
                "recovery_point": {
                    "iteration": checkpoint.iteration,
                    "best_score": checkpoint.best_score,
                    "progress": checkpoint.progress_percentage
                },
                "recommendations": self._generate_recovery_recommendations(checkpoint)
            }
            
            logger.info(f"Recovery possible from iteration {checkpoint.iteration} "
                       f"(best score: {checkpoint.best_score:.3f})")
            
            return recovery_info
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return None
    
    def _validate_checkpoint(self, checkpoint: CheckpointData) -> bool:
        """Validate checkpoint data integrity"""
        try:
            # Check required fields
            required_fields = ['timestamp', 'iteration', 'current_prompt', 'best_score']
            for field in required_fields:
                if not hasattr(checkpoint, field) or getattr(checkpoint, field) is None:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Check data types and ranges
            if checkpoint.iteration < 0:
                logger.error("Invalid iteration number")
                return False
            
            if not (0.0 <= checkpoint.best_score <= 1.0):
                logger.error("Invalid best score range")
                return False
            
            if not checkpoint.current_prompt.strip():
                logger.error("Empty current prompt")
                return False
            
            # Check timestamp is reasonable (not too old or in future)
            current_time = time.time()
            if checkpoint.timestamp > current_time + 3600:  # 1 hour in future
                logger.error("Checkpoint timestamp is in the future")
                return False
            
            if current_time - checkpoint.timestamp > 7 * 24 * 3600:  # 1 week old
                logger.warning("Checkpoint is more than 1 week old")
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation error: {e}")
            return False
    
    def _generate_recovery_recommendations(self, checkpoint: CheckpointData) -> List[str]:
        """Generate recommendations for recovery"""
        recommendations = []
        
        # Check error count
        if checkpoint.error_count > 10:
            recommendations.append("High error count detected. Consider reviewing error patterns.")
        
        # Check progress
        if checkpoint.progress_percentage < 20:
            recommendations.append("Low progress. Consider adjusting optimization strategy.")
        elif checkpoint.progress_percentage > 80:
            recommendations.append("High progress achieved. Focus on fine-tuning.")
        
        # Check best score
        if checkpoint.best_score < 0.5:
            recommendations.append("Low best score. Consider fundamental prompt improvements.")
        elif checkpoint.best_score > 0.8:
            recommendations.append("High score achieved. Focus on consistency and edge cases.")
        
        # Check last error
        if checkpoint.last_error:
            if "rate limit" in checkpoint.last_error.lower():
                recommendations.append("Rate limit errors detected. Consider reducing API call frequency.")
            elif "parsing" in checkpoint.last_error.lower():
                recommendations.append("Parsing errors detected. Review response format handling.")
        
        # Check checkpoint age
        age_hours = (time.time() - checkpoint.timestamp) / 3600
        if age_hours > 24:
            recommendations.append(f"Checkpoint is {age_hours:.1f} hours old. Verify current relevance.")
        
        return recommendations