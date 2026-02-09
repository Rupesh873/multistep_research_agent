#verification.py
def verification_node(state):
    warnings = []

    if not state.get("web_search"):
        warnings.append("No web data")
    if not state.get("content_reader"):
        warnings.append("No content read")
    if not state.get("authority_source"):
        warnings.append("No authority sources")

    return {
        "verification": {
            "status": "OK" if not warnings else "Partial",
            "warnings": warnings
        }
    }
