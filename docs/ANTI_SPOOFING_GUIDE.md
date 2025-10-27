# Anti-Spoofing Troubleshooting Guide

## Common Issues with Voice Recognition and Anti-Spoofing

### "Replay Attack Detected" False Positives

If you're getting "REPLAY ATTACK DETECTED" warnings when recording your own voice in a quiet room, try these solutions:

1. **Enable Development Mode**: 
   ```yaml
   # In config.yaml
   system:
     development_mode: true
   ```
   This relaxes anti-spoofing checks during testing and development.

2. **Bypass Anti-Spoofing (for testing only)**:
   ```yaml
   # In config.yaml
   system:
     bypass_anti_spoofing: true
   ```
   WARNING: Only use this option for testing, never in production.

3. **Adjust Your Environment**:
   - Move to a room with some ambient noise (completely silent rooms can trigger false positives)
   - Position yourself 6-12 inches from the microphone
   - Speak at a normal, conversational volume
   
4. **Check Your Audio Setup**:
   - Use a good quality microphone if available
   - Avoid using built-in laptop microphones which can have unusual frequency response
   - Disable any audio processing/effects on your microphone

5. **Adjust Anti-Spoofing Thresholds**:
   ```yaml
   # In config.yaml - Increase these values to make detection less sensitive
   security:
     min_security_confidence: 0.5  # (default 0.6, higher = less sensitive)
   ```

## How Anti-Spoofing Works

The system checks for signs that your audio might be:

1. **Replayed** (recorded and played back)
   - Missing high frequency content
   - Unnatural phase patterns
   - Double compression artifacts

2. **Synthetic** (computer-generated)
   - Overly smooth pitch contours
   - Unnatural formant transitions
   - Missing micro-variations in voice

In quiet environments with certain microphones, these checks might be too sensitive. The system has been updated to better handle quiet environments, but you may still need to use the options above in some cases.

## Production Environments

For production deployment:
- Always set `development_mode: false`
- Always set `bypass_anti_spoofing: false`
- Ensure users are in a suitable environment for voice recording
- Consider training staff on proper microphone usage