from db.connection import transaction
from db.ids import new_id

def create_tenant(display_name):
    tid = new_id()
    with transaction() as cur:
        cur.execute("INSERT INTO tenants (tenant_id, display_name) VALUES (%s, %s)",
                    (tid, display_name))
    return tid

def create_property(tenant_id, kind, display_name):     # kind: 'stay' | 'cafe'
    pid = new_id()
    with transaction() as cur:
        cur.execute("""INSERT INTO properties (property_id, tenant_id, kind, display_name)
                       VALUES (%s, %s, %s, %s)""", (pid, tenant_id, kind, display_name))
    return pid

def create_room(property_id, display_name, slug):
    rid = new_id()
    with transaction() as cur:
        cur.execute("""INSERT INTO rooms (room_id, property_id, display_name, slug)
                       VALUES (%s, %s, %s, %s)""", (rid, property_id, display_name, slug))
    return rid