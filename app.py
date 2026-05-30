import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


# 强制指定文泉驿字体 + 兜底英文字体
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
# 解决负号显示为方框
matplotlib.rcParams['axes.unicode_minus'] = False
# 全局禁用缓存（避免旧字体配置残留）
matplotlib.rcParams['backend'] = 'Agg'



import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import sqlite3
import hashlib
import os
import secrets
from datetime import datetime

# ---------------------------
# 基础配置
# ---------------------------
st.set_page_config(page_title="多元统计分析平台（含登录与用户管理）", layout="wide")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False

DB_PATH = "app_users.db"

# ---------------------------
# DB & Auth Utils
# ---------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT NOT NULL,
        detail TEXT,
        ts TEXT NOT NULL
    )
    """)
    conn.commit()

    # 初始化一个默认管理员（如果不存在）
    c.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    admin_count = c.fetchone()[0]
    if admin_count == 0:
         #默认管理员：admin / Admin@12345（你可在后台改掉）
        create_user("admin", "Admin@12345", role="admin", conn=conn)
    conn.close()

def hash_password(password: str, salt: str) -> str:
    # PBKDF2-HMAC 更好；这里用 sha256+多轮也可。毕业设计建议 PBKDF2。
    # 这里直接用 pbkdf2_hmac：
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 120_000)
    return dk.hex()

def create_user(username: str, password: str, role="user", conn=None):
    close_after = False
    if conn is None:
        conn = get_conn()
        close_after = True
    c = conn.cursor()

    salt = secrets.token_hex(16)
    ph = hash_password(password, salt)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO users(username, password_hash, salt, role, is_active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
        (username, ph, salt, role, created_at)
    )
    conn.commit()
    if close_after:
        conn.close()

def verify_user(username: str, password: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT username, password_hash, salt, role, is_active FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return False, None
    u, ph, salt, role, is_active = row
    if is_active != 1:
        return False, None
    return hash_password(password, salt) == ph, {"username": u, "role": role}

def log_action(username: str, action: str, detail: str = ""):
    conn = get_conn()
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO audit_logs(username, action, detail, ts) VALUES (?, ?, ?, ?)",
              (username, action, detail, ts))
    conn.commit()
    conn.close()

def require_login():
    return st.session_state.get("auth", {}).get("is_login", False)

def current_user():
    return st.session_state.get("auth", {}).get("user")

def is_admin():
    u = current_user()
    return bool(u) and u.get("role") == "admin"

def logout():
    st.session_state["auth"] = {"is_login": False, "user": None}
    # 清理数据也可以按需做（这里不强制）
    # st.session_state.pop("df", None)

# ---------------------------
# Data Utils
# ---------------------------
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def numeric_df(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number])

def data_profile(df: pd.DataFrame) -> pd.DataFrame:
    # 简易数据画像表
    prof = []
    for col in df.columns:
        s = df[col]
        prof.append({
            "字段名": col,
            "类型": str(s.dtype),
            "缺失数": int(s.isna().sum()),
            "缺失率": float(s.isna().mean()),
            "唯一值数": int(s.nunique(dropna=True)),
            "样例(前3)": ", ".join([str(x) for x in s.dropna().head(3).tolist()])
        })
    return pd.DataFrame(prof)

def detect_outliers_iqr(df_num: pd.DataFrame, k=1.5):
    outlier_mask = pd.DataFrame(False, index=df_num.index, columns=df_num.columns)
    for col in df_num.columns:
        x = df_num[col].dropna()
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        outlier_mask.loc[df_num.index, col] = (df_num[col] < low) | (df_num[col] > high)
    return outlier_mask

def detect_outliers_zscore(df_num: pd.DataFrame, z=3.0):
    outlier_mask = pd.DataFrame(False, index=df_num.index, columns=df_num.columns)
    for col in df_num.columns:
        x = df_num[col]
        mu, sigma = x.mean(), x.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            continue
        zs = (x - mu) / sigma
        outlier_mask[col] = zs.abs() > z
    return outlier_mask

# ---------------------------
# UI: Auth Pages
# ---------------------------
def auth_page():
    st.title("🔐 多元统计分析平台 - 登录/注册")

    tabs = st.tabs(["登录", "注册", "说明"])
    with tabs[0]:
        username = st.text_input("用户名", key="login_user")
        password = st.text_input("密码", type="password", key="login_pwd")
        if st.button("登录", use_container_width=True):
            ok, info = verify_user(username, password)
            if ok:
                st.session_state["auth"] = {"is_login": True, "user": info}
                log_action(username, "LOGIN", "用户登录成功")
                st.success("登录成功！请在左侧选择功能。")
                st.rerun()
            else:
                log_action(username or "UNKNOWN", "LOGIN_FAIL", "登录失败/账户禁用")
                st.error("登录失败：用户名/密码错误，或账户被禁用。")

        st.caption("默认管理员账号：admin / Admin@12345（首次运行自动创建，建议登录后立刻修改密码）")

    with tabs[1]:
        st.subheader("注册新用户（普通用户）")
        nu = st.text_input("新用户名", key="reg_user")
        npw = st.text_input("新密码", type="password", key="reg_pwd")
        npw2 = st.text_input("确认密码", type="password", key="reg_pwd2")
        if st.button("注册", use_container_width=True):
            if not nu or not npw:
                st.warning("用户名和密码不能为空。")
            elif npw != npw2:
                st.warning("两次密码不一致。")
            elif len(npw) < 8:
                st.warning("密码建议至少 8 位。")
            else:
                try:
                    create_user(nu, npw, role="user")
                    log_action(nu, "REGISTER", "注册成功")
                    st.success("注册成功！请返回登录。")
                except sqlite3.IntegrityError:
                    st.error("该用户名已存在。")

    with tabs[2]:
        st.markdown("""
- 本平台支持：数据上传清洗、相关性分析、PCA、聚类、多元回归，并扩展了异常值检测、特征筛选、模型评估与用户管理。
- 管理员可在「用户管理」中禁用/启用用户、重置密码、查看审计日志。
        """)

# ---------------------------
# UI: Admin User Management
# ---------------------------
def admin_user_management():
    st.header("👥 用户管理（管理员）")

    conn = get_conn()
    c = conn.cursor()

    st.subheader("创建用户（管理员可创建 admin/user）")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_u = st.text_input("用户名", key="admin_new_user")
    with col2:
        new_p = st.text_input("初始密码", type="password", key="admin_new_pwd")
    with col3:
        new_r = st.selectbox("角色", ["user", "admin"], key="admin_new_role")

    if st.button("创建用户", type="primary"):
        try:
            create_user(new_u, new_p, role=new_r, conn=conn)
            log_action(current_user()["username"], "ADMIN_CREATE_USER", f"create {new_u} role={new_r}")
            st.success("创建成功。")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("用户名已存在。")
        except Exception as e:
            st.error(f"创建失败：{e}")

    st.divider()

    st.subheader("用户列表")
    users = pd.read_sql_query("SELECT id, username, role, is_active, created_at FROM users ORDER BY id DESC", conn)
    st.dataframe(users, use_container_width=True)

    st.divider()

    st.subheader("用户操作")
    user_names = users["username"].tolist()
    target = st.selectbox("选择用户", user_names, key="admin_target_user")
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("启用用户"):
            c.execute("UPDATE users SET is_active=1 WHERE username=?", (target,))
            conn.commit()
            log_action(current_user()["username"], "ADMIN_ENABLE_USER", target)
            st.success("已启用。")
            st.rerun()

    with action_col2:
        if st.button("禁用用户"):
            if target == "admin":
                st.warning("不建议禁用默认管理员。")
            else:
                c.execute("UPDATE users SET is_active=0 WHERE username=?", (target,))
                conn.commit()
                log_action(current_user()["username"], "ADMIN_DISABLE_USER", target)
                st.success("已禁用。")
                st.rerun()

    with action_col3:
        reset_pwd = st.text_input("重置密码为", type="password", key="admin_reset_pwd")
        if st.button("重置密码"):
            if not reset_pwd or len(reset_pwd) < 8:
                st.warning("新密码至少 8 位。")
            else:
                salt = secrets.token_hex(16)
                ph = hash_password(reset_pwd, salt)
                c.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?", (ph, salt, target))
                conn.commit()
                log_action(current_user()["username"], "ADMIN_RESET_PASSWORD", target)
                st.success("密码已重置。")
                st.rerun()

    st.divider()

    st.subheader("审计日志（最近 200 条）")
    logs = pd.read_sql_query("SELECT username, action, detail, ts FROM audit_logs ORDER BY id DESC LIMIT 200", conn)
    st.dataframe(logs, use_container_width=True)

    conn.close()

# ---------------------------
# UI: Main App
# ---------------------------
def main_app():
    # 侧边栏导航
    st.sidebar.title("📊 统计分析系统")
    st.sidebar.caption(f"当前用户：{current_user()['username']}（{current_user()['role']}）")

    if st.sidebar.button("退出登录"):
        log_action(current_user()["username"], "LOGOUT", "用户退出")
        logout()
        st.rerun()

    menu = [
        "数据上传与预处理",
        "数据概览与质量",
        "相关性分析",
        "异常值检测",
        "特征选择",
        "PCA降维",
        "聚类分析",
        "多元回归分析",
        "导出与报告"
    ]
    if is_admin():
        menu.insert(0, "用户管理")

    choice = st.sidebar.selectbox("功能导航", menu)

    # 管理功能
    if choice == "用户管理":
        admin_user_management()
        return

    # 需要数据的功能：检查 df
    def require_df():
        if "df" not in st.session_state or st.session_state["df"] is None:
            st.warning("请先在「数据上传与预处理」上传数据。")
            return False
        return True

    # --- 1. 数据上传与预处理 ---
    if choice == "数据上传与预处理":
        st.header("📂 数据导入与清洗")
        uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])

        if uploaded_file:
            df = load_data(uploaded_file)
            st.session_state["df"] = df
            log_action(current_user()["username"], "UPLOAD_DATA", f"file={uploaded_file.name}, shape={df.shape}")

            st.subheader("数据预览")
            st.dataframe(df.head(20), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"行数: {df.shape[0]} | 列数: {df.shape[1]}")
                if st.checkbox("显示缺失值统计"):
                    st.write(df.isnull().sum())

            with col2:
                st.subheader("数据清洗")
                fill_method = st.selectbox("缺失值处理", ["不处理", "均值填充", "中位数填充", "删除缺失行"])
                drop_dup = st.checkbox("删除重复行", value=False)

                if st.button("执行清洗", type="primary"):
                    df2 = df.copy()
                    if drop_dup:
                        df2 = df2.drop_duplicates()

                    if fill_method == "均值填充":
                        df2 = df2.fillna(df2.mean(numeric_only=True))
                    elif fill_method == "中位数填充":
                        df2 = df2.fillna(df2.median(numeric_only=True))
                    elif fill_method == "删除缺失行":
                        df2 = df2.dropna()

                    st.session_state["df"] = df2
                    log_action(current_user()["username"], "CLEAN_DATA", f"method={fill_method}, drop_dup={drop_dup}")
                    st.success("清洗完成！已更新当前数据。")

    # --- 数据概览与质量 ---
    elif choice == "数据概览与质量":
        st.header("🧾 数据概览与质量报告")
        if not require_df():
            return
        df = st.session_state["df"]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("字段画像（Profiling）")
            prof = data_profile(df)
            st.dataframe(prof, use_container_width=True)
        with col2:
            st.subheader("总体信息")
            st.write({
                "行数": df.shape[0],
                "列数": df.shape[1],
                "数值列数": numeric_df(df).shape[1],
                "缺失单元格总数": int(df.isna().sum().sum())
            })

        st.subheader("描述统计（数值列）")
        st.dataframe(numeric_df(df).describe().T, use_container_width=True)

        st.subheader("缺失率可视化（数值列）")
        miss = df.isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss) == 0:
            st.info("没有缺失值。")
        else:
            fig = px.bar(miss, title="各字段缺失率", labels={"index": "字段", "value": "缺失率"})
            st.plotly_chart(fig, use_container_width=True)

    # --- 2. 相关性分析 ---
    elif choice == "相关性分析":
        st.header("🔍 相关性矩阵与散点矩阵")
        if not require_df():
            return
        df = numeric_df(st.session_state["df"]).dropna()

        if df.shape[1] < 2:
            st.warning("数值列不足，无法进行相关性分析。")
            return

        st.subheader("相关性热力图")
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig_corr)

        st.subheader("交互式散点矩阵 (Plotly)")
        selected_cols = st.multiselect(
            "选择绘图变量（建议 3-6 个）",
            df.columns.tolist(),
            default=df.columns.tolist()[:min(3, df.shape[1])]
        )
        if selected_cols and len(selected_cols) >= 2:
            fig_scatter = px.scatter_matrix(df[selected_cols])
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 异常值检测 ---
    elif choice == "异常值检测":
        st.header("🚨 异常值检测")
        if not require_df():
            return
        df = numeric_df(st.session_state["df"])

        if df.shape[1] == 0:
            st.warning("没有数值列可用于异常值检测。")
            return

        method = st.selectbox("方法", ["IQR", "Z-score"])
        if method == "IQR":
            k = st.slider("IQR 系数 k", 1.0, 3.0, 1.5, 0.1)
            mask = detect_outliers_iqr(df, k=k)
        else:
            z = st.slider("Z 阈值", 2.0, 5.0, 3.0, 0.1)
            mask = detect_outliers_zscore(df, z=z)

        outlier_counts = mask.sum().sort_values(ascending=False)
        st.subheader("各字段异常值数量")
        st.dataframe(outlier_counts.to_frame("异常值数量"), use_container_width=True)

        st.subheader("异常值占比（Top 15）")
        ratio = (mask.mean()).sort_values(ascending=False).head(15)
        fig = px.bar(ratio, title="异常值占比 Top 15", labels={"index": "字段", "value": "异常值占比"})
        st.plotly_chart(fig, use_container_width=True)

        st.info("提示：异常值并不一定是错误数据；可结合业务背景决定是否处理。")

    # --- 特征选择 ---
    elif choice == "特征选择":
        st.header("🧩 特征选择（数值列）")
        if not require_df():
            return
        df0 = numeric_df(st.session_state["df"]).copy()

        if df0.shape[1] < 2:
            st.warning("数值列不足。")
            return

        st.subheader("方法 1：方差阈值筛选")
        var_th = st.slider("方差阈值", 0.0, float(df0.var().max() if df0.var().max() > 0 else 1.0), 0.0)
        keep_var = df0.var() >= var_th
        kept_cols_var = df0.columns[keep_var].tolist()
        st.write(f"保留 {len(kept_cols_var)} / {df0.shape[1]} 列：", kept_cols_var)

        st.subheader("方法 2：相关性去冗余（与其它特征相关系数过高则剔除）")
        corr_th = st.slider("相关性阈值（绝对值）", 0.5, 0.99, 0.9, 0.01)
        corr = df0[kept_cols_var].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_th)]
        final_cols = [c for c in kept_cols_var if c not in to_drop]

        st.write("建议剔除：", to_drop if to_drop else "无")
        st.success(f"最终建议保留 {len(final_cols)} 列：{final_cols}")

        if st.button("将筛选后的数据保存为当前数据（仅保留这些数值列）", type="primary"):
            # 注意：这里只保留筛选后的数值列；如果你希望保留原非数值列，可改为 df_all.join(...)
            st.session_state["df"] = st.session_state["df"][final_cols].copy()
            log_action(current_user()["username"], "FEATURE_SELECT", f"final_cols={len(final_cols)}")
            st.success("已更新当前数据。")
            st.rerun()

    # --- 3. PCA 降维 ---
    elif choice == "PCA降维":
        st.header("📉 主成分分析 (PCA)")
        if not require_df():
            return
        df = numeric_df(st.session_state["df"]).dropna()

        if df.shape[1] < 2:
            st.warning("数值列不足，无法 PCA。")
            return

        max_comp = min(10, df.shape[1])
        n_components = st.slider("选择主成分数量", 2, max(2, max_comp), 2)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)

        st.write(f"累计解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")

        var_df = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(n_components)],
            "解释方差比": pca.explained_variance_ratio_
        })
        st.dataframe(var_df, use_container_width=True)

        pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", title="PCA 2D 投影")
        st.plotly_chart(fig_pca, use_container_width=True)

        # 载荷（贡献）
        loadings = pd.DataFrame(pca.components_.T, index=df.columns,
                                columns=[f"PC{i+1}" for i in range(n_components)])
        st.subheader("主成分载荷矩阵（Loadings）")
        st.dataframe(loadings, use_container_width=True)

    # --- 4. 聚类分析 ---
    elif choice == "聚类分析":
        st.header("🧪 聚类分析 (K-means + 层次聚类)")
        if not require_df():
            return
        df = numeric_df(st.session_state["df"]).dropna()

        if df.shape[1] < 2:
            st.warning("数值列不足。")
            return

        st.subheader("K-means")
        k = st.sidebar.slider("选择 K 值", 2, 12, 3)

        # 肘部法则
        with st.expander("查看肘部法则（WCSS）", expanded=False):
            max_k = st.slider("计算到的最大K", 3, 15, 10)
            wcss = []
            X = df.values
            for kk in range(1, max_k + 1):
                km = KMeans(n_clusters=kk, random_state=42, n_init="auto")
                km.fit(X)
                wcss.append(km.inertia_)
            fig_elbow = px.line(x=list(range(1, max_k + 1)), y=wcss, markers=True,
                                title="Elbow Method (WCSS)", labels={"x": "K", "y": "WCSS"})
            st.plotly_chart(fig_elbow, use_container_width=True)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(df)
        df_plot = df.copy()
        df_plot["Cluster"] = clusters

        x_col, y_col = df.columns[0], df.columns[1]
        fig_cluster = px.scatter(df_plot, x=x_col, y=y_col, color="Cluster", title="K-means 聚类结果")
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.subheader("聚类中心（Cluster Centers）")
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)
        st.dataframe(centers, use_container_width=True)

        # 层次聚类
        if st.checkbox("显示层次聚类树状图"):
            st.subheader("层次聚类 (Dendrogram)")
            use_cols = df.columns[:min(8, df.shape[1])]
            fig_dendro, ax = plt.subplots(figsize=(10, 5))
            Z = linkage(df[use_cols], "ward")
            dendrogram(Z, ax=ax)
            ax.set_title("Hierarchical Clustering Dendrogram")
            st.pyplot(fig_dendro)

    # --- 5. 多元回归分析 ---
    elif choice == "多元回归分析":
        st.header("📈 多元线性回归（含评估）")
        if not require_df():
            return
        df = numeric_df(st.session_state["df"]).dropna()

        if df.shape[1] < 2:
            st.warning("数值列不足。")
            return

        all_cols = df.columns.tolist()
        y_var = st.selectbox("选择因变量 (Y)", all_cols)
        x_vars = st.multiselect("选择自变量 (X)", [c for c in all_cols if c != y_var])

        if x_vars:
            test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            X = df[x_vars]
            y = df[y_var]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # statsmodels（可解释性强）
            X_train_sm = sm.add_constant(X_train)
            model = sm.OLS(y_train, X_train_sm).fit()

            st.subheader("回归摘要（训练集）")
            st.text(model.summary())

            # 评估（测试集）
            X_test_sm = sm.add_constant(X_test, has_constant="add")
            y_pred = model.predict(X_test_sm)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.subheader("测试集评估指标")
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2:.4f}")
            c2.metric("MAE", f"{mae:.4f}")
            c3.metric("RMSE", f"{rmse:.4f}")

            st.subheader("预测 vs 真值")
            pv = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred.values})
            fig_pv = px.scatter(pv, x="y_true", y="y_pred", title="Predicted vs Actual")
            st.plotly_chart(fig_pv, use_container_width=True)

            st.subheader("残差分析图")
            resid = y_test.values - y_pred.values
            fig_res, ax = plt.subplots(figsize=(7, 4))
            sns.scatterplot(x=y_pred.values, y=resid, ax=ax)
            ax.axhline(0, linestyle="--")
            ax.set_title("Residuals vs Fitted (Test Set)")
            ax.set_xlabel("Fitted")
            ax.set_ylabel("Residual")
            st.pyplot(fig_res)

            log_action(current_user()["username"], "REGRESSION_RUN",
                       f"y={y_var}, x={x_vars}, test_size={test_size}")

    # --- 导出与报告 ---
    elif choice == "导出与报告":
        st.header("📦 导出与报告")
        if not require_df():
            return
        df = st.session_state["df"]

        st.subheader("下载当前数据")
        st.download_button(
            label="下载当前数据 (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="current_data.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.subheader("下载数据摘要（Profiling + describe）")
        prof = data_profile(df)
        desc = numeric_df(df).describe().T.reset_index().rename(columns={"index": "字段"})
        # 合并成一个 excel 更好，但为了简单输出两个 CSV
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "下载 Profiling (CSV)",
                data=prof.to_csv(index=False).encode("utf-8"),
                file_name="profiling.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "下载 Describe (CSV)",
                data=desc.to_csv(index=False).encode("utf-8"),
                file_name="describe.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()

        st.subheader("一键导出“分析简报”（兼容你原逻辑）")
        if st.button("生成并导出 analysis_report.csv", type="primary"):
            # 简单：直接导出当前数据
            log_action(current_user()["username"], "EXPORT_REPORT", "analysis_report.csv")
            st.download_button(
                label="点击下载 analysis_report.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="analysis_report.csv",
                mime="text/csv",
                use_container_width=True
            )

# ---------------------------
# App Entry
# ---------------------------
def bootstrap_session():
    if "auth" not in st.session_state:
        st.session_state["auth"] = {"is_login": False, "user": None}
    if "df" not in st.session_state:
        st.session_state["df"] = None

def main():
    init_db()
    bootstrap_session()

    if not require_login():
        auth_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
