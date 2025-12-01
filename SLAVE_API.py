import psycopg2
from psycopg2 import pool
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
import time
import threading
import sys
from datetime import datetime

# DB Configuration
DB_MASTER_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

DB_SLAVE_CONFIG = {
    "host": "localhost",
    "port": 5434,
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}


# Failover and Connection Management
class DBManager:
    """Manages connections and failover."""

    # Read traffic will prefer Slave but fall back to Master if Slave is unavailable.
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
        try:
            temp_conn = psycopg2.connect(**DB_SLAVE_CONFIG)
            temp_conn.close()
            return True
        except Exception:
            return False

    def _health_monitor(self):
        """Checks slave status and manages the slave connection pool."""
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
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SLAVE DOWN. Read traffic affected.")

            time.sleep(self.HEALTH_CHECK_INTERVAL)

    def get_read_connection(self):
        """Gets a connection for reads, using failover logic."""

        if DBManager.FORCE_SLAVE_ONLY_READS:
            # SLAVE-ONLY MODE
            if not DBManager.is_slave_available or not DBManager.slave_pool:
                raise HTTPException(
                    status_code=503,
                    detail="Read Service Unavailable: Slave-only mode is active and Slave DB is unavailable. Reads are blocked."
                )
            try:
                conn = DBManager.slave_pool.getconn()
                return conn, "SLAVE (FORCED MODE)"
            except Exception:
                with DBManager.state_lock:
                    DBManager.is_slave_available = False
                raise HTTPException(
                    status_code=503,
                    detail="Read Service Unavailable: Slave connection failed. Forced mode active."
                )

        else:
            # FAILOVER MODE
            if DBManager.is_slave_available and DBManager.slave_pool:
                try:
                    conn = DBManager.slave_pool.getconn()
                    return conn, "SLAVE"
                except Exception:
                    with DBManager.state_lock:
                        DBManager.is_slave_available = False
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SLAVE FAILURE on use. Falling back to MASTER.")

            # Use Master (failover)
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


# API Schema and Models
app = FastAPI(
    title="Partitioned Financial Transaction API with Failover",
    description="Provides real-time analytics from a partitioned PostgreSQL database (trn) with automatic Master/Slave failover.",
    version="1.0.0"
)


class PersonInfo(BaseModel):
    pt_key: int
    first_name: str
    last_name: str
    person_id: str


class AccountSummary(BaseModel):
    acc_key: int
    acc_num: str
    total_credit: float
    total_debit: float


class AggregationResult(BaseModel):
    aggregate_value: float
    query_type: str


class TopAccount(BaseModel):
    acc_num: str
    transaction_count: int


class MonthlySummary(BaseModel):
    transaction_month: str
    transaction_count: int
    total_amount: float


# API Endpoints

# READ ENDPOINTS (Use Slave/Master Failover)

@app.get("/system/status", summary="Check current read database status", response_model=Dict[str, Any])
def get_system_status():
    """Shows which database (Slave or Master) is currently handling read traffic."""

    if DBManager.FORCE_SLAVE_ONLY_READS:
        read_source = "SLAVE (FORCED MODE)"
    else:
        read_source = "SLAVE" if DBManager.is_slave_available else "MASTER (Failover)"

    status = {
        "mode": "SLAVE_ONLY" if DBManager.FORCE_SLAVE_ONLY_READS else "FAILOVER",
        "is_slave_available": DBManager.is_slave_available,
        "read_source": read_source,
        "last_checked": datetime.now().isoformat()
    }
    return status


@app.get("/stats/monthly_summary", summary="Get monthly transaction volume and count",
         response_model=List[MonthlySummary])
def get_monthly_transaction_summary(db_info: tuple = Depends(get_read_db)):
    """Calculates monthly transaction volume and count for the last 12 months."""
    conn, db_type = db_info
    cur = conn.cursor()

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

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


@app.get("/stats/transaction_average", summary="Get the average transaction amount", response_model=AggregationResult)
def get_average_transaction(db_info: tuple = Depends(get_read_db)):
    """Calculates the average transaction amount across all transactions."""
    conn, db_type = db_info
    cur = conn.cursor()

    try:
        cur.execute("SELECT AVG(trn_amt) FROM trn;")
        avg_amount = cur.fetchone()[0]

        return {
            "aggregate_value": float(avg_amount) if avg_amount else 0.0,
            "query_type": f"AVG(trn_amt) from {db_type}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


@app.get("/analytics/top_accounts", summary="Get the Top N accounts by transaction volume",
         response_model=List[TopAccount])
def get_top_n_accounts_by_transactions(n: int = 10, db_info: tuple = Depends(get_read_db)):
    """Returns the top N accounts by total transaction volume."""
    conn, db_type = db_info
    cur = conn.cursor()

    try:
        cur.execute(f"""
            WITH AccountActivity AS (
                SELECT cr_acc_key AS acc_key FROM trn
                UNION ALL
                SELECT db_acc_key AS acc_key FROM trn
            )
            SELECT 
                t1.acc_num, 
                COUNT(t2.acc_key) AS transaction_count
            FROM 
                acc t1
            JOIN
                AccountActivity t2 ON t1.acc_key = t2.acc_key
            GROUP BY 
                t1.acc_num
            ORDER BY 
                transaction_count DESC
            LIMIT {n};
        """)

        results = [{"acc_num": row[0], "transaction_count": row[1]} for row in cur.fetchall()]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


@app.get("/query/person/{person_id}/accounts", summary="Get accounts for a specific person",
         response_model=List[AccountSummary])
def get_person_accounts(person_id: str, db_info: tuple = Depends(get_read_db)):
    """Finds a person and lists their accounts with balance summaries."""
    conn, db_type = db_info
    cur = conn.cursor()

    try:
        cur.execute("SELECT pt_key FROM pt WHERE person_id = %s;", (person_id,))
        pt_key_result = cur.fetchone()

        if not pt_key_result:
            raise HTTPException(status_code=404, detail=f"Person with ID '{person_id}' not found.")

        target_pt_key = pt_key_result[0]

        cur.execute(f"""
            SELECT
                a.acc_key,
                a.acc_num,
                COALESCE(SUM(CASE WHEN t.cr_acc_key = a.acc_key THEN t.trn_amt ELSE 0 END), 0) AS total_credit,
                COALESCE(SUM(CASE WHEN t.db_acc_key = a.acc_key THEN t.trn_amt ELSE 0 END), 0) AS total_debit
            FROM
                acc a
            LEFT JOIN
                trn t ON a.acc_key = t.cr_acc_key OR a.acc_key = t.db_acc_key
            WHERE
                a.pt_key = %s
            GROUP BY
                a.acc_key, a.acc_num
            ORDER BY
                a.acc_key;
        """, (target_pt_key,))

        results = []
        for row in cur.fetchall():
            results.append({
                "acc_key": row[0],
                "acc_num": row[1],
                "total_credit": float(row[2]),
                "total_debit": float(row[3])
            })

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


# WRITE ENDPOINT (Use Master Only)

class NewTransaction(BaseModel):
    cr_acc_num: str
    db_acc_num: str
    amount: float
    description: str


@app.post("/transactions/record", summary="Record a new transaction (Write to Master)", response_model=Dict[str, str])
def record_transaction(transaction: NewTransaction, db_info: tuple = Depends(get_write_db)):
    """Records a new transaction."""
    conn, db_type = db_info
    cur = conn.cursor()

    try:
        cur.execute("""
                    SELECT acc_key, ccy_key
                    FROM acc
                    WHERE acc_num IN (%s, %s);
                    """, (transaction.cr_acc_num, transaction.db_acc_num))

        accounts = cur.fetchall()

        if len(accounts) < 2:
            raise HTTPException(status_code=400, detail="One or both account numbers are invalid.")

        acc_map = {acc_num: (acc_key, ccy_key) for acc_key, ccy_key in accounts}

        cr_acc_key, trn_ccy_key = acc_map.get(transaction.cr_acc_num, (None, None))
        db_acc_key, _ = acc_map.get(transaction.db_acc_num, (None, None))

        if cr_acc_key == db_acc_key:
            raise HTTPException(status_code=400, detail="Credit and Debit accounts cannot be the same.")

        trn_date = datetime.now()
        cur.execute("""
                    INSERT INTO trn (trn_date, cr_acc_key, db_acc_key, trn_ccy_key, trn_amt, description)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING trn_id;
                    """, (trn_date, cr_acc_key, db_acc_key, trn_ccy_key, transaction.amount, transaction.description))

        new_trn_id = cur.fetchone()[0]
        conn.commit()

        return {
            "message": "Transaction recorded successfully.",
            "transaction_id": str(new_trn_id),
            "written_to": db_type,
            "partitioned_date": trn_date.isoformat()
        }

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error on {db_type}: {str(e)}")
    finally:
        cur.close()


if __name__ == "__main__":
    import uvicorn

    print("\n--- Starting High Availability API (FAILOVER MODE ACTIVE) ---")
    print(f"Read Mode: {'SLAVE-ONLY (Active)' if DBManager.FORCE_SLAVE_ONLY_READS else 'MASTER/SLAVE Failover'}")
    print("Master is assumed to be on 5433, Slave on 5434.")
    print("Background health check is running...")
    uvicorn.run(app, host="127.0.0.1", port=8000)