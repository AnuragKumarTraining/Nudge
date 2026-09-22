from psycopg2.extras import Json
from db.connection import transaction

def create_capture(capture_id, room_id, captured_at, s3_prefix):
    with transaction() as cur:
        cur.execute("""INSERT INTO captures (capture_id, room_id, status, captured_at, s3_prefix)
                       VALUES (%s, %s, 'PENDING', %s, %s)""",
                    (capture_id, room_id, captured_at, s3_prefix))

def claim_capture(capture_id):
    """Atomic claim. Returns the row, or None if another worker already has it."""
    with transaction() as cur:
        cur.execute("""UPDATE captures SET status = 'PROCESSING'
                       WHERE capture_id = %s AND status = 'PENDING'
                       RETURNING room_id, s3_prefix""", (capture_id,))
        return cur.fetchone()

def mark_rejected(capture_id, reason):
    with transaction() as cur:
        cur.execute("UPDATE captures SET status = 'REJECTED', reject_reason = %s WHERE capture_id = %s",
                    (reason, capture_id))

def mark_complete(capture_id, master_id, ssim, brightness_delta,
                  findings, vlm_verdict, checklist):
    with transaction() as cur:
        cur.execute("""UPDATE captures
                       SET status = 'COMPLETE', master_id = %s, ssim = %s,
                           brightness_delta = %s, findings = %s, vlm_verdict = %s,
                           checklist = %s, processed_at = now()
                       WHERE capture_id = %s""",
                    (master_id, ssim, brightness_delta, Json(findings),
                     vlm_verdict, Json(checklist), capture_id))