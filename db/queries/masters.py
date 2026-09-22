from db.connection import transaction
from db.ids import new_id

def create_master(room_id, s3_prefix, variant="default"):
    mid = new_id()
    with transaction() as cur:
        cur.execute("""INSERT INTO masters (master_id, room_id, variant, status, s3_prefix)
                       VALUES (%s, %s, %s, 'PENDING', %s)""",
                    (mid, room_id, variant, s3_prefix))
    return mid

def get_active_master(room_id, variant="default"):
    with transaction() as cur:
        cur.execute("""SELECT * FROM masters
                       WHERE room_id = %s AND variant = %s AND status = 'ACTIVE'""",
                    (room_id, variant))
        return cur.fetchone()          # None if the room has no active master

def activate_master(master_id):
    """Retire the current ACTIVE master, then activate this one, in one transaction."""
    with transaction() as cur:
        cur.execute("SELECT room_id, variant FROM masters WHERE master_id = %s FOR UPDATE",
                    (master_id,))
        m = cur.fetchone()
        cur.execute("""UPDATE masters SET status = 'RETIRED'
                       WHERE room_id = %s AND variant = %s
                         AND status = 'ACTIVE' AND master_id <> %s""",
                    (m["room_id"], m["variant"], master_id))
        cur.execute("""UPDATE masters SET status = 'ACTIVE', activated_at = now()
                       WHERE master_id = %s""", (master_id,))

def reject_master(master_id, reason):
    with transaction() as cur:
        cur.execute("UPDATE masters SET status = 'REJECTED', reject_reason = %s WHERE master_id = %s",
                    (reason, master_id))