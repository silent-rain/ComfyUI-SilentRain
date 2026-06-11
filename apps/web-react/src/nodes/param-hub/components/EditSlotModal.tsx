import React, { useMemo, useState, useEffect } from 'react';
import type { Slot } from '../../../types/comfy';
import { useParamHubStore } from '../../../store';
import styles from './EditSlotModal.module.scss';

interface EditSlotModalProps {
  nodeId: number;
  onClose: () => void;
}

/**
 * 批量编辑 Slot Label 的弹窗组件
 * - 支持批量编辑所有 slots 的 label
 * - 检查重复 label 并显示红框提醒
 */
export const EditSlotModal: React.FC<EditSlotModalProps> = ({ nodeId, onClose }) => {
  const [slotData, setSlotData] = useState<Slot[]>([]);

  // 直接订阅 store 的 hubs Map，hubs 引用变化时触发重渲染
  const hubs = useParamHubStore();
  console.log(`[EditSlotModal] Hubs nodeId: ${nodeId}, `, hubs.getHubSlots(nodeId));

  // 当 hubs 或 nodeId 变化时，同步 slotData
  useEffect(() => {
    const slots = hubs.getHubSlots(nodeId);
    setSlotData(slots);
  }, [hubs, nodeId]);

  // 检查 label 是否重复
  const labelErrors = useMemo(() => {
    const labelCounts: { [key: string]: number } = {};
    const errors: { [key: string]: boolean } = {};

    // 统计每个 label 出现的次数
    slotData.forEach(slot => {
      const label = (slot.label ?? '').trim();
      if (label) {
        labelCounts[label] = (labelCounts[label] || 0) + 1;
      }
    });

    // 标记重复的 label
    slotData.forEach(slot => {
      const label = (slot.label ?? '').trim();
      if (label && (labelCounts[label] || 0) > 1) {
        errors[slot.name] = true;
      } else {
        errors[slot.name] = false;
      }
    });

    return errors;
  }, [slotData]);

  const hasErrors = useMemo(() => Object.values(labelErrors).some(error => error), [labelErrors]);

  const handleLabelChange = (name: string, value: string) => {
    setSlotData(prev => prev.map(slot => (slot.name === name ? { ...slot, label: value } : slot)));
  };

  const handleSave = () => {
    for (const slot of slotData) {
      hubs.updateSlotLabel(nodeId, slot.name, slot.label);
    }
    console.log(`[EditSlotModal] Hubs nodeId: ${nodeId}  save, `, hubs.getHubSlots(nodeId));
    onClose();
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>Edit Slot Labels</span>
          <button className={styles.closeBtn} onClick={onClose}>
            ×
          </button>
        </div>

        <div className={styles.body}>
          <div className={styles.slotList}>
            {slotData.map(slot => (
              <div key={slot.name} className={styles.slotItem}>
                <div className={styles.slotInfo}>
                  <span className={styles.slotId}>{slot.name}</span>
                  <span className={styles.slotType}>{slot.type}</span>
                </div>
                <div className={styles.field}>
                  <input
                    className={`${styles.input} ${labelErrors[slot.name] ? styles.inputError : ''}`}
                    type='text'
                    value={slot.label}
                    onChange={e => handleLabelChange(slot.name, e.target.value)}
                    placeholder='Enter slot label...'
                  />
                  {labelErrors[slot.name] && (
                    <div className={styles.errorMsg}>
                      ⚠ Label already exists! Please use a unique label.
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className={styles.footer}>
          <button className={styles.cancelBtn} onClick={onClose}>
            Cancel
          </button>
          <button className={styles.saveBtn} onClick={handleSave} disabled={hasErrors}>
            Save All
          </button>
        </div>
      </div>
    </div>
  );
};
