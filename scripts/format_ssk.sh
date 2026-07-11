#!/bin/bash
# 格式化 SSK 256GB 移动硬盘为 ext4（整个磁盘）
# 警告：这会清除所有数据！

set -e

DEVICE="/dev/sda"

echo "⚠️  即将擦除 $DEVICE 上的所有数据！"
echo "设备信息："
lsblk "$DEVICE" -o NAME,SIZE,TYPE,MODEL
echo ""
read -p "确认继续？(输入 YES 继续): " confirm

if [ "$confirm" != "YES" ]; then
    echo "已取消"
    exit 1
fi

# 1. 卸载
echo ">>> 卸载分区..."
sudo umount "${DEVICE}1" 2>/dev/null || true
sudo umount "${DEVICE}2" 2>/dev/null || true
sudo umount "${DEVICE}3" 2>/dev/null || true

# 2. 创建 GPT 分区表 + 单分区
echo ">>> 创建分区表..."
echo -e "g\nn\n\n\n\nw" | sudo fdisk "$DEVICE"

# 3. 格式化为 ext4
echo ">>> 格式化为 ext4..."
sudo mkfs.ext4 -L "SSK-STORAGE" "${DEVICE}1"

# 4. 验证
echo ""
echo ">>> 验证结果："
sudo partprobe "$DEVICE" 2>/dev/null || true
lsblk "$DEVICE" -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT
