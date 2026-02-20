"""
Multi-Level Alert Engine
Determines alert level and triggers appropriate actions
"""

import sys
import os
from datetime import datetime
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AlertLevel(Enum):
    """Alert severity levels"""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    ANOMALY = 3


class AlertEngine:
    def __init__(self, 
                 warning_threshold=75,
                 critical_threshold=85,
                 anomaly_threshold=95,
                 forecast_warning_threshold=80,
                 forecast_critical_threshold=90):
        """
        Initialize alert engine with thresholds
        
        Args:
            warning_threshold: Trigger warning at this % usage
            critical_threshold: Trigger critical at this % usage
            anomaly_threshold: Trigger anomaly at this % usage
            forecast_warning_threshold: Trigger warning if forecast exceeds this
            forecast_critical_threshold: Trigger critical if forecast exceeds this
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.anomaly_threshold = anomaly_threshold
        self.forecast_warning_threshold = forecast_warning_threshold
        self.forecast_critical_threshold = forecast_critical_threshold
        
        # Alert history
        self.alert_history = []
    
    def analyze_prediction(self, prediction_result):
        """
        Analyze prediction and determine alert level
        
        Args:
            prediction_result: Dict from RealtimePredictor.predict_from_dataframe()
        
        Returns:
            dict with alert level and details
        """
        current = prediction_result['current_values']
        
        # Initialize alert info
        alert_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': AlertLevel.NORMAL,
            'severity_score': 0.0,
            'reasons': [],
            'recommended_actions': [],
            'current_values': current,
            'predicted_class': None,
            'confidence': 0.0
        }
        
        # Check classification if available
        if 'classification' in prediction_result:
            class_info = prediction_result['classification']
            predicted_class = class_info['class_name']
            confidence = class_info['confidence']
            
            alert_info['predicted_class'] = predicted_class
            alert_info['confidence'] = confidence
            
            # Direct anomaly detection from classifier
            if predicted_class == 'Anomaly' and confidence > 0.6:
                alert_info['level'] = AlertLevel.ANOMALY
                alert_info['severity_score'] = 1.0
                alert_info['reasons'].append(f"Model classified as ANOMALY (confidence: {confidence*100:.1f}%)")
                alert_info['recommended_actions'].append("⚠️  IMMEDIATE ACTION REQUIRED")
                alert_info['recommended_actions'].append("🛑 Consider killing high-resource processes")
                alert_info['recommended_actions'].append("📊 Review system logs for unusual activity")
                return alert_info
            
            elif predicted_class == 'Critical' and confidence > 0.5:
                alert_info['level'] = AlertLevel.CRITICAL
                alert_info['severity_score'] = max(alert_info['severity_score'], 0.8)
                alert_info['reasons'].append(f"Model classified as CRITICAL (confidence: {confidence*100:.1f}%)")
            
            elif predicted_class == 'Warning' and confidence > 0.5:
                alert_info['level'] = AlertLevel.WARNING
                alert_info['severity_score'] = max(alert_info['severity_score'], 0.5)
                alert_info['reasons'].append(f"Model classified as WARNING (confidence: {confidence*100:.1f}%)")
        
        # Check current values
        max_current = max(current['cpu'], current['memory'], current['disk'])
        
        if max_current >= self.anomaly_threshold:
            alert_info['level'] = AlertLevel.ANOMALY
            alert_info['severity_score'] = 1.0
            for metric, value in current.items():
                if value >= self.anomaly_threshold:
                    alert_info['reasons'].append(f"{metric.upper()} at {value:.1f}% (ANOMALY threshold: {self.anomaly_threshold}%)")
        
        elif max_current >= self.critical_threshold:
            alert_info['level'] = AlertLevel.CRITICAL
            alert_info['severity_score'] = max(alert_info['severity_score'], 0.8)
            for metric, value in current.items():
                if value >= self.critical_threshold:
                    alert_info['reasons'].append(f"{metric.upper()} at {value:.1f}% (CRITICAL threshold: {self.critical_threshold}%)")
        
        elif max_current >= self.warning_threshold:
            if alert_info['level'] == AlertLevel.NORMAL:
                alert_info['level'] = AlertLevel.WARNING
                alert_info['severity_score'] = max(alert_info['severity_score'], 0.5)
            for metric, value in current.items():
                if value >= self.warning_threshold:
                    alert_info['reasons'].append(f"{metric.upper()} at {value:.1f}% (WARNING threshold: {self.warning_threshold}%)")
        
        # Check forecast if available
        if 'forecast' in prediction_result:
            forecast = prediction_result['forecast']
            
            # Check if any forecast value exceeds thresholds
            for metric in ['cpu', 'memory', 'disk']:
                forecast_values = forecast[metric]
                max_forecast = max(forecast_values)
                
                if max_forecast >= self.forecast_critical_threshold:
                    if alert_info['level'].value < AlertLevel.CRITICAL.value:
                        alert_info['level'] = AlertLevel.CRITICAL
                        alert_info['severity_score'] = max(alert_info['severity_score'], 0.8)
                    alert_info['reasons'].append(
                        f"Predicted {metric.upper()} spike to {max_forecast:.1f}% in next 25 seconds"
                    )
                
                elif max_forecast >= self.forecast_warning_threshold:
                    if alert_info['level'].value < AlertLevel.WARNING.value:
                        alert_info['level'] = AlertLevel.WARNING
                        alert_info['severity_score'] = max(alert_info['severity_score'], 0.5)
                    alert_info['reasons'].append(
                        f"Predicted {metric.upper()} increase to {max_forecast:.1f}% in next 25 seconds"
                    )
                
                # Check trend (rapid increase)
                if len(forecast_values) >= 2:
                    trend = forecast_values[-1] - forecast_values[0]
                    if trend > 15:  # More than 15% increase
                        alert_info['reasons'].append(
                            f"Rapid {metric.upper()} increase trend detected (+{trend:.1f}%)"
                        )
        
        # Add recommended actions based on alert level
        if alert_info['level'] == AlertLevel.ANOMALY:
            alert_info['recommended_actions'].extend([
                "🛑 SYSTEM CRASH IMMINENT - Take immediate action",
                "🔴 Kill high-resource processes immediately",
                "📞 Notify system administrator",
                "💾 Consider emergency system restart"
            ])
        elif alert_info['level'] == AlertLevel.CRITICAL:
            alert_info['recommended_actions'].extend([
                "⚠️  HIGH PRIORITY - Resource usage critical",
                "🔍 Identify and stop resource-intensive processes",
                "📊 Check for memory leaks or runaway processes",
                "🔔 Escalate to on-call engineer"
            ])
        elif alert_info['level'] == AlertLevel.WARNING:
            alert_info['recommended_actions'].extend([
                "⚡ Monitor closely - Resources trending upward",
                "🔎 Review recent system changes",
                "📈 Check application logs for anomalies",
                "🕐 Prepare for potential intervention"
            ])
        else:
            alert_info['recommended_actions'].append("✅ System operating normally")
        
        # Add to history
        self.alert_history.append(alert_info)
        
        return alert_info
    
    def print_alert(self, alert_info):
        """Pretty print alert information"""
        level = alert_info['level']
        
        # Icons and colors
        level_icons = {
            AlertLevel.NORMAL: '✅',
            AlertLevel.WARNING: '⚠️ ',
            AlertLevel.CRITICAL: '🔴',
            AlertLevel.ANOMALY: '☠️ '
        }
        
        level_names = {
            AlertLevel.NORMAL: 'NORMAL',
            AlertLevel.WARNING: 'WARNING',
            AlertLevel.CRITICAL: 'CRITICAL',
            AlertLevel.ANOMALY: 'ANOMALY'
        }
        
        print("\n" + "="*60)
        print(f"{level_icons[level]} ALERT: {level_names[level]}")
        print("="*60)
        
        print(f"\n🕐 Timestamp: {alert_info['timestamp']}")
        print(f"📊 Severity Score: {alert_info['severity_score']*100:.0f}%")
        
        if alert_info['predicted_class']:
            print(f"🤖 AI Classification: {alert_info['predicted_class']} ({alert_info['confidence']*100:.1f}% confidence)")
        
        print(f"\n📈 Current System Status:")
        print(f"   CPU:    {alert_info['current_values']['cpu']:.1f}%")
        print(f"   Memory: {alert_info['current_values']['memory']:.1f}%")
        print(f"   Disk:   {alert_info['current_values']['disk']:.1f}%")
        
        if alert_info['reasons']:
            print(f"\n⚠️  Reasons:")
            for reason in alert_info['reasons']:
                print(f"   • {reason}")
        
        print(f"\n💡 Recommended Actions:")
        for action in alert_info['recommended_actions']:
            print(f"   {action}")
        
        print("\n" + "="*60 + "\n")
    
    def should_crash_system(self, alert_info):
        """
        Determine if system should crash based on alert
        
        Returns:
            bool: True if system should crash
        """
        return alert_info['level'] == AlertLevel.ANOMALY
    
    def get_alert_summary(self, last_n=10):
        """Get summary of recent alerts"""
        recent_alerts = self.alert_history[-last_n:]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_level': {
                'Normal': 0,
                'Warning': 0,
                'Critical': 0,
                'Anomaly': 0
            }
        }
        
        for alert in recent_alerts:
            level_name = alert['level'].name.capitalize()
            if level_name == 'Normal':
                level_name = 'Normal'
            summary['by_level'][level_name] += 1
        
        return summary


def test_alert_engine():
    """Test alert engine with sample predictions"""
    from ai_model.predict import RealtimePredictor
    import pandas as pd
    
    print("🧪 Testing Alert Engine...\n")
    
    # Initialize alert engine
    engine = AlertEngine()
    
    # Test case 1: Normal
    print("\n📋 Test Case 1: Normal Usage")
    sample_normal = pd.DataFrame({
        'cpu_usage': [30 + i for i in range(12)],
        'memory_usage': [35 + i for i in range(12)],
        'disk_usage': [50] * 12
    })
    
    # Test case 2: Warning
    print("\n📋 Test Case 2: Warning - Gradual Increase")
    sample_warning = pd.DataFrame({
        'cpu_usage': [30 + i*4 for i in range(12)],
        'memory_usage': [35 + i*3 for i in range(12)],
        'disk_usage': [50] * 12
    })
    
    # Test case 3: Critical
    print("\n📋 Test Case 3: Critical - High Usage")
    sample_critical = pd.DataFrame({
        'cpu_usage': [70 + i*2 for i in range(12)],
        'memory_usage': [75 + i for i in range(12)],
        'disk_usage': [55] * 12
    })
    
    # Test case 4: Anomaly
    print("\n📋 Test Case 4: Anomaly - Extreme Usage")
    sample_anomaly = pd.DataFrame({
        'cpu_usage': [85 + i for i in range(12)],
        'memory_usage': [90 + i*0.5 for i in range(12)],
        'disk_usage': [95] * 12
    })
    
    test_cases = [
        ("Normal", sample_normal),
        ("Warning", sample_warning),
        ("Critical", sample_critical),
        ("Anomaly", sample_anomaly)
    ]
    
    predictor = RealtimePredictor()
    
    for name, data in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        # Make prediction
        prediction = predictor.predict_from_dataframe(data)
        
        # Analyze with alert engine
        alert = engine.analyze_prediction(prediction)
        engine.print_alert(alert)
        
        if engine.should_crash_system(alert):
            print("🛑 SYSTEM CRASH TRIGGERED!")


if __name__ == "__main__":
    test_alert_engine()