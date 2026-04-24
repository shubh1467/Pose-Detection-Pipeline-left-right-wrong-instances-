# 🏏 Cricket Pose Classification System

A real-time computer vision system that classifies a cricket batsman's stance as **FRONT VIEW**, **SIDE VIEW**, or **WRONG INSTANCE** using YOLOv11x-pose and custom bat detection.

---

## 📌 **Overview**

This system analyzes video footage of cricket batsmen and determines their body orientation using pose keypoints and facial features. The classification helps in understanding batsman's stance, batting technique, and body positioning during play.

### **Key Features**
- ✅ Real-time pose estimation using YOLOv11x-pose
- ✅ Bat detection and tracking
- ✅ Front/Side view classification with high accuracy
- ✅ Visual feedback with colored bounding boxes
- ✅ Temporal smoothing for stable predictions
- ✅ Debug mode with feature visualization

---

## 🎯 **Classification Outputs**

| Label | Color | Description |
|-------|-------|-------------|
| **FRONT VIEW** | 🟢 Green | Batsman facing the camera directly |
| **SIDE VIEW** | 🟠 Orange | Batsman in profile (side-on stance) |
| **WRONG INSTANCE** | 🔴 Red | Poor quality frame or ambiguous pose |

---

## 🏗️ **System Architecture**
