# AI Agent Working Guide - Project Starlight

**Working in an isolated sandbox environment.**

## Context

- **Project**: Steganography detection in blockchain images
- **Goal**: Safeguard integrity of digital history on-chain
- **Location**: `/data/uploads/results/[visible_pixel_hash]/` - your workspace

## Security Constraints

### Isolation
- Working dir: sandbox only
- File access: sandbox only
- Network: no external access

## Task Workflow

1. Read `memory.md` first for context
2. Implement solution
3. Test locally
4. Update `memory.md` with new state

### Standard Response Format
```python
{
    "success": True,
    "result": {...},
    "error": None,
    "metadata": {"task_completed": True, "completion_time": datetime.datetime.now().isoformat()}
}
```

## Deliverables
- Implementation code/logic
- Evidence of completion (tests, outputs)
- Working files in sandbox
- Summary (status, results)
- Web content: Single Page Application, navigation, visualizations

### SPA Development
- Use `npm`, `vite`, `webpack`, `rollup` for building
- Use [sql.js](https://sql.js.org/) for client-side SQLite
- Use WebRTC for peer-to-peer features
- **Do NOT keep Node.js server running** - Stargate serves static files

```javascript
// WebRTC example
const pc = new RTCPeerConnection({iceServers: [{urls: 'stun:stun.l.google.com:19302'}]});
pc.ondatachannel = e => {
    const dc = e.channel;
    dc.onmessage = m => console.log('Received:', m.data);
    dc.send('Hello!');
};
```

## Visualization

### Chart.js
```python
{"type": "bar", "data": {"labels": ["Clean", "Stego"], "datasets": [{"data": [95, 82]}]}}
```

### Plotly
```python
{"type": "heatmap", "z": data, "x": ["256", "512"], "y": ["LSB", "Alpha"]}
```

## Work Completion

1. Test everything
2. Document code
3. Submit deliverables

### Submission Format
```python
{
    "notes": "# Task Report\n## Implementation\n...",
    "result_file": "/uploads/results/[hash]/[task_id].md",
    "artifacts_dir": "/uploads/results/[hash]/",
    "completion_proof": "unique-id",
    "web_content": {"html_files": [], "data": []}
}
```

## Checklist
- [ ] Code runs without errors
- [ ] Security constraints respected
- [ ] All deliverables present
- [ ] Documentation provided
- [ ] Evidence of completion

## Reminders
- Be concise and technical
- Focus on implementation details
- Provide concrete evidence
- Never access files outside sandbox

---
**Ready to work!**
