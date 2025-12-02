import psycopg2
from psycopg2 import pool
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
import time
import threading
import sys
from datetime import datetime

# =================================================================
# DB CONFIGURATION AND CONNECTION MANAGEMENT
# =================================================================

DB_MASTER_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": "postgres",
    "connect_timeout": 3
}

DB_SLAVE_CONFIG = {
    "host": "localhost",
    "port": 5434,
    "database": "postgres",
    "user": "postgres",
    "password": "postgres",
    "connect_timeout": 3
}


class DBManager:
    """Manages connections and failover."""

    FORCE_SLAVE_ONLY_READS = False
    is_slave_available = False
    master_pool = None
    slave_pool = None

    state_lock = threading.Lock()
    HEALTH_CHECK_INTERVAL = 10

    def __init__(self):
        if not DBManager.master_pool:
            try:
                DBManager.master_pool = pool.SimpleConnectionPool(1, 5, **DB_MASTER_CONFIG)
                print("MASTER Connection Pool Initialized.")
            except Exception as e:
                print(f"FATAL: Could not connect to Master DB: {e}")
                sys.exit(1)

        threading.Thread(target=self._health_monitor, daemon=True).start()

    def _check_slave_health(self):
        """Attempts a quick connection to the slave."""
        try:
            temp_conn = psycopg2.connect(**DB_SLAVE_CONFIG)
            temp_conn.close()
            return True
        except Exception:
            return False

    def _health_monitor(self):
        """Continuously monitors slave status."""
        while True:
            is_healthy = self._check_slave_health()

            with DBManager.state_lock:
                if is_healthy and not DBManager.is_slave_available:
                    try:
                        DBManager.slave_pool = pool.SimpleConnectionPool(1, 5, **DB_SLAVE_CONFIG)
                        DBManager.is_slave_available = True
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] SLAVE RECOVERED. Reads switching back.")
                    except Exception:
                        DBManager.is_slave_available = False

                elif not is_healthy and DBManager.is_slave_available:
                    DBManager.is_slave_available = False
                    if DBManager.slave_pool:
                        DBManager.slave_pool.closeall()
                        DBManager.slave_pool = None
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SLAVE DOWN. Read traffic falling back to MASTER.")

            time.sleep(self.HEALTH_CHECK_INTERVAL)

    def get_read_connection(self):
        """Gets a connection for reads, prioritizing Slave."""
        if DBManager.is_slave_available and DBManager.slave_pool:
            try:
                conn = DBManager.slave_pool.getconn()
                return conn, "SLAVE"
            except Exception:
                with DBManager.state_lock:
                    DBManager.is_slave_available = False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SLAVE FAILURE on use. Falling back to MASTER.")

        try:
            conn = DBManager.master_pool.getconn()
            return conn, "MASTER (Failover)"
        except Exception as e:
            print(f"FATAL: MASTER connection also failed: {e}")
            raise HTTPException(status_code=503, detail="Database Service Unavailable (Master Failure)")

    def get_write_connection(self):
        """Gets a connection for writes (Master only)."""
        try:
            conn = DBManager.master_pool.getconn()
            return conn, "MASTER"
        except Exception as e:
            print(f"FATAL: MASTER connection failed: {e}")
            raise HTTPException(status_code=503, detail="Database Service Unavailable (Master Failure)")

    def release_connection(self, conn, db_type):
        """Releases the connection back to the appropriate pool."""
        if "SLAVE" in db_type and DBManager.slave_pool:
            DBManager.slave_pool.putconn(conn)
        elif "MASTER" in db_type and DBManager.master_pool:
            DBManager.master_pool.putconn(conn)


db_manager = DBManager()


def get_read_db():
    conn, db_type = db_manager.get_read_connection()
    try:
        yield conn, db_type
    finally:
        db_manager.release_connection(conn, db_type)


def get_write_db():
    conn, db_type = db_manager.get_write_connection()
    try:
        yield conn, db_type
    finally:
        db_manager.release_connection(conn, db_type)


# =================================================================
# API SCHEMA AND MODELS
# =================================================================
app = FastAPI(
    title="Partitioned Financial Transaction API with Failover",
    description="Provides real-time analytics from a partitioned PostgreSQL database with automatic Master/Slave failover.",
    version="1.0.0"
)


class PersonInfo(BaseModel):
    pt_key: int
    first_name: str
    last_name: str
    person_id: str
    birth_date: Optional[datetime] = None


class AccountSummary(BaseModel):
    acc_key: int
    acc_num: str
    total_credit: float
    total_debit: float


class AggregationResult(BaseModel):
    aggregate_value: float
    query_type: str
    execution_time_ms: str


class MonthlySummary(BaseModel):
    transaction_month: str
    transaction_count: int
    total_amount: float


class NewTransaction(BaseModel):
    cr_acc_num: str
    db_acc_num: str
    amount: float
    ccy_key_char: str
    description: str


def log_performance(endpoint: str, db_type: str, start_time: float, target_ms: int = 1000, is_write: bool = False):
    end_time = time.perf_counter()
    exec_time_ms = (end_time - start_time) * 1000

    db_info = f" | {db_type}" if db_type else ""
    op_type = "Write" if is_write else "Read"

    status = "✓" if exec_time_ms < target_ms else "✗ SLOW"
    print(f"[{endpoint}]{db_info} | {op_type} Time: {exec_time_ms:.2f}ms (Target: <{target_ms}ms) {status}")

    # Warning if query exceeds target
    if exec_time_ms >= target_ms:
        print(f"  WARNING: Query exceeded {target_ms}ms target!")

    return exec_time_ms


# =================================================================
# READ ENDPOINTS (Use Slave/Master Failover)
# =================================================================

@app.get("/system/status", summary="Check current read database status", response_model=Dict[str, Any])
def get_system_status():
    """Shows which database (Slave or Master) is currently handling read traffic."""
    start_time = time.perf_counter()
    read_source = "SLAVE" if DBManager.is_slave_available else "MASTER (Failover)"

    exec_time_ms = log_performance("/system/status", "", start_time)

    status = {
        "is_slave_available": DBManager.is_slave_available,
        "read_source": read_source,
        "last_checked": datetime.now().isoformat(),
        "execution_time_ms": f"{exec_time_ms:.2f}"
    }
    return status


@app.get("/stats/monthly_summary", summary="Get monthly transaction volume and count",
         response_model=Dict[str, Any])
def get_monthly_transaction_summary(db_info: tuple = Depends(get_read_db)):
    """Calculates monthly transaction volume and count for the last 12 months."""
    conn, db_type = db_info
    cur = conn.cursor()
    start_time = time.perf_counter()

    try:
        cur.execute("""
                    SELECT TO_CHAR(DATE_TRUNC('month', trn_date), 'YYYY-MM') AS transaction_month,
                           COUNT(trn_id)                                     AS transaction_count,
                           SUM(trn_amt)                                      AS total_amount
                    FROM trn
                    WHERE trn_date >= (NOW() - INTERVAL '1 year')
                    GROUP BY 1
                    ORDER BY 1 DESC;
                    """)

        results = []
        for row in cur.fetchall():
            results.append({
                "transaction_month": row[0],
                "transaction_count": row[1],
                "total_amount": float(row[2])
            })

        exec_time_ms = log_performance("/stats/monthly_summary", db_type, start_time)

        return {
            "data": results,
            "execution_time_ms": f"{exec_time_ms:.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


@app.get("/stats/transaction_average", summary="Get the average transaction amount", response_model=AggregationResult)
def get_average_transaction(months: int = 3, db_info: tuple = Depends(get_read_db)):
    """Calculates the average transaction amount for recent months (default: last 3 months).
    Optimized with time-based filtering to ensure <100ms response time.
    """
    conn, db_type = db_info
    cur = conn.cursor()
    start_time = time.perf_counter()

    try:
        # Only scan recent partitions for faster results
        cur.execute("""
            SELECT AVG(trn_amt) 
            FROM trn 
            WHERE trn_date >= (NOW() - INTERVAL '%s months');
        """ % months)
        avg_amount = cur.fetchone()[0]

        exec_time_ms = log_performance("/stats/transaction_average", db_type, start_time, target_ms=100)

        return {
            "aggregate_value": float(avg_amount) if avg_amount else 0.0,
            "query_type": f"AVG(trn_amt) last {months} months from {db_type}",
            "execution_time_ms": f"{exec_time_ms:.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


@app.get("/query/person/{person_id}/accounts", summary="Get accounts for a specific person",
         response_model=Dict[str, Any])
def get_person_accounts(person_id: str, db_info: tuple = Depends(get_read_db)):
    """Finds a person and lists their accounts with balance summaries (optimized for partitions)."""
    conn, db_type = db_info
    cur = conn.cursor()
    start_time = time.perf_counter()

    try:
        cur.execute("SELECT pt_key FROM pt WHERE person_id = %s;", (person_id,))
        pt_key_result = cur.fetchone()

        if not pt_key_result:
            exec_time_ms = log_performance("/query/person/{person_id}/accounts", db_type, start_time)
            raise HTTPException(status_code=404, detail=f"Person with ID '{person_id}' not found.")

        target_pt_key = pt_key_result[0]

        sql_query = """
                    SELECT a.acc_key, \
                           a.acc_num, \
                           COALESCE(Credit.credit_sum, 0) AS total_credit, \
                           COALESCE(Debit.debit_sum, 0)   AS total_debit
                    FROM acc a
                             LEFT JOIN
                         (SELECT cr_acc_key, SUM(trn_amt) AS credit_sum
                          FROM trn
                          WHERE cr_acc_key IN (SELECT acc_key FROM acc WHERE pt_key = %s)
                          GROUP BY cr_acc_key) Credit
                         ON a.acc_key = Credit.cr_acc_key
                             LEFT JOIN
                         (SELECT db_acc_key, SUM(trn_amt) AS debit_sum
                          FROM trn
                          WHERE db_acc_key IN (SELECT acc_key FROM acc WHERE pt_key = %s)
                          GROUP BY db_acc_key) Debit
                         ON a.acc_key = Debit.db_acc_key
                    WHERE a.pt_key = %s
                    ORDER BY a.acc_key; \
                    """
        cur.execute(sql_query, (target_pt_key, target_pt_key, target_pt_key))

        results = []
        for row in cur.fetchall():
            results.append({
                "acc_key": row[0],
                "acc_num": row[1],
                "total_credit": float(row[2]),
                "total_debit": float(row[3])
            })

        exec_time_ms = log_performance("/query/person/{person_id}/accounts", db_type, start_time)

        return {
            "person_id": person_id,
            "accounts": results,
            "execution_time_ms": f"{exec_time_ms:.2f}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


# =================================================================
# WRITE ENDPOINT (Use Master Only)
# =================================================================

@app.post("/transactions/record", summary="Record a new transaction (Write to Master)", response_model=Dict[str, str])
def record_transaction(transaction: NewTransaction, db_info: tuple = Depends(get_write_db)):
    """Records a new transaction."""
    conn, db_type = db_info
    cur = conn.cursor()
    start_time = time.perf_counter()

    try:
        cur.execute("""
                    SELECT acc_key, ccy_key, acc_num
                    FROM acc
                    WHERE acc_num IN (%s, %s);
                    """, (transaction.cr_acc_num, transaction.db_acc_num))

        accounts = cur.fetchall()

        if len(accounts) < 2:
            raise HTTPException(status_code=400, detail="One or both account numbers are invalid.")

        acc_map = {acc_num: (acc_key, ccy_key) for acc_key, ccy_key, acc_num in accounts}

        cr_acc_key, cr_ccy_key_int = acc_map.get(transaction.cr_acc_num, (None, None))
        db_acc_key, db_ccy_key_int = acc_map.get(transaction.db_acc_num, (None, None))

        if cr_acc_key is None or db_acc_key is None:
            raise HTTPException(status_code=400, detail="One or both account numbers are invalid or not found.")

        if cr_acc_key == db_acc_key:
            raise HTTPException(status_code=400, detail="Credit and Debit accounts cannot be the same.")

        cur.execute("""
                    SELECT ccy_int_key
                    FROM ccy
                    WHERE ccy_key = %s;
                    """, (transaction.ccy_key_char,))

        trn_ccy_key_int_result = cur.fetchone()

        if not trn_ccy_key_int_result:
            raise HTTPException(status_code=400,
                                detail=f"Currency code '{transaction.ccy_key_char}' is not a valid currency key.")

        trn_ccy_key_int = trn_ccy_key_int_result[0]

        trn_date = datetime.now()
        cur.execute("""
                    INSERT INTO trn (trn_date, cr_acc_key, db_acc_key, trn_ccy_key, trn_amt, description)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING trn_id;
                    """,
                    (trn_date, cr_acc_key, db_acc_key, trn_ccy_key_int, transaction.amount, transaction.description))

        new_trn_id = cur.fetchone()[0]
        conn.commit()

        exec_time_ms = log_performance("/transactions/record", db_type, start_time, is_write=True)

        return {
            "message": "Transaction recorded successfully.",
            "transaction_id": str(new_trn_id),
            "written_to": db_type,
            "partitioned_date": trn_date.isoformat(),
            "execution_time_ms": f"{exec_time_ms:.2f}"
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"Transaction insertion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


if __name__ == "__main__":
    import uvicorn

    print("\n--- Starting High Availability API (MASTER/SLAVE FAILOVER) ---")
    print("Master is assumed to be on 5433, Slave on 5434.")
    print("Run the SQL schema first and ensure your two PostgreSQL instances are running.")
    print("Background health check is running...")
    print("Performance Target: All queries must complete in <1000ms")
    uvicorn.run(app, host="127.0.0.1", port=8000)