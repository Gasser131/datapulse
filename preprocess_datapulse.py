# -*- coding: utf-8 -*-
# تنظيف وتجهيز بيانات DataPulse لنموذج التعلم الآلي

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "DataPulse_Operations_Data.csv"
OUTPUT_FILE = "DataPulse_Cleaned_ML_Ready.csv"

# ---------------------------------------------------
# 1) قراءة البيانات
# ---------------------------------------------------
df = pd.read_csv(INPUT_FILE)
print("الأبعاد الأصلية:", df.shape)
print("\nالقيم المفقودة قبل المعالجة:")
print(df.isna().sum())

df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------------------------------
# 2) إزالة التكرارات
# ---------------------------------------------------
dups = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nعدد السجلات المكررة اللي اتحذفت: {dups}")

# ---------------------------------------------------
# 3) القيم المفقودة
# بنعوّض بالوسيط (median) لكل نوع خدمة، لأن سعر الخدمة واستقرارها
# بيختلفوا من خدمة لأخرى، والوسيط مش بيتأثر بالقيم الشاذة
# ---------------------------------------------------
for col in ["Revenue_USD", "Server_Uptime_Pct"]:
    missing = df[col].isna().sum()
    df[col] = df[col].fillna(df.groupby("Service_Type")[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())   # احتياطي لو خدمة كلها فاضية
    print(f"{col}: تم تعويض {missing} قيمة")

# ---------------------------------------------------
# 4) القيم الشاذة (طريقة IQR)
# بنكتشفها وبنحدّها بأقرب حد مقبول (capping) بدل ما نحذف الصف،
# لأن عدد السجلات قليل أصلاً
# ---------------------------------------------------
print("\nالقيم الشاذة:")
for col in ["Revenue_USD", "Server_Uptime_Pct", "Support_Tickets"]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[col] < low) | (df[col] > high)]
    print(f"{col}: {len(outliers)} قيمة شاذة", outliers["Record_ID"].tolist())
    df[col] = df[col].clip(low, high)

# نسبة الاستقرار لازم تفضل بين 0 و 100
df["Server_Uptime_Pct"] = df["Server_Uptime_Pct"].clip(0, 100)
# الإيراد وعدد التذاكر مينفعش يبقوا سالبين
df["Revenue_USD"] = df["Revenue_USD"].clip(lower=0)
df["Support_Tickets"] = df["Support_Tickets"].clip(lower=0)

# ---------------------------------------------------
# 5) هندسة الخصائص (من التاريخ والعلاقات التشغيلية)
# ---------------------------------------------------
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Day_Of_Week"] = df["Date"].dt.dayofweek
df["Is_Weekend"] = (df["Day_Of_Week"] >= 5).astype(int)
df["Days_Since_Start"] = (df["Date"] - df["Date"].min()).dt.days
df["Downtime_Pct"] = 100 - df["Server_Uptime_Pct"]
df["Revenue_Per_Ticket"] = df["Revenue_USD"] / (df["Support_Tickets"] + 1)

# ---------------------------------------------------
# 6) ترميز الأعمدة النصية (One-Hot)
# ---------------------------------------------------
df = pd.get_dummies(df, columns=["Service_Type", "Status"], dtype=int)

# الأعمدة دي مش بتدخل النموذج (معرّف / تاريخ خام / اسم نصي)
df = df.drop(columns=["Record_ID", "Date", "Client_Name"])

# ---------------------------------------------------
# 7) تطبيع الأعمدة الرقمية (StandardScaler)
# ---------------------------------------------------
num_cols = ["Revenue_USD", "Server_Uptime_Pct", "Support_Tickets",
            "Downtime_Pct", "Revenue_Per_Ticket", "Days_Since_Start",
            "Month", "Day", "Day_Of_Week"]
df[num_cols] = StandardScaler().fit_transform(df[num_cols])

# ---------------------------------------------------
# 8) فحص أخير وحفظ الملف
# ---------------------------------------------------
print("\nقيم مفقودة بعد المعالجة:", df.isna().sum().sum())
print("الأبعاد النهائية:", df.shape)

df.to_csv(OUTPUT_FILE, index=False)
print("تم حفظ الملف:", OUTPUT_FILE)
