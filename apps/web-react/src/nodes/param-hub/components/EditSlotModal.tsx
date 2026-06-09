import React, { useState, useEffect, useCallback } from 'react';
import { useParamHubStore } from '../../../store';
import styles from './EditSlotModal.module.scss';

interface EditSlotModalProps {
  nodeId: number;
  onClose: () => void;
}

interface SlotData {
  linkId: number;
  currentName: string;
  type: string;
}

/**
 * 批量编辑 Slot 名称的弹窗组件
 * - 支持批量编辑所有 slots 的名称
 * - 检查重复名称并显示红框提醒
 */
export const EditSlotModal: React.FC<EditSlotModalProps> = ({ nodeId, onClose }) => {
  const getHubSlots = useParamHubStore(state => state.getHubSlots);
  const updateSlotLabel = useParamHubStore(state => state.updateSlotLabel);

  const slots = getHubSlots(nodeId);
  const [slotData, setSlotData] = useState<SlotData[]>([]);
  const [nameErrors, setNameErrors] = useState<{ [key: number]: boolean }>({});

  // 初始化 slot 数据
  useEffect(() => {
    if (slots && slots.size > 0) {
      const data: SlotData[] = Array.from(slots.entries()).map(([linkId, slot]) => ({
        linkId,
        currentName: slot.label ?? slot.name ?? '',
        type: String(slot.type),
      }));
      setSlotData(data);
    }
  }, [slots]);

  // 检查名称是否重复
  const checkDuplicates = useCallback((slotData: SlotData[]) => {
    const nameCounts: { [key: string]: number } = {};
    const errors: { [key: number]: boolean } = {};

    // 统计每个名称出现的次数
    slotData.forEach(slot => {
      const name = slot.currentName.trim();
      if (name) {
        nameCounts[name] = (nameCounts[name] || 0) + 1;
      }
    });

    // 标记重复的名称
    slotData.forEach(slot => {
      const name = slot.currentName.trim();
      if (name && (nameCounts[name] || 0) > 1) {
        errors[slot.linkId] = true;
      } else {
        errors[slot.linkId] = false;
      }
    });

    return errors;
  }, []);

  // 实时检查重复
  useEffect(() => {
    const errors = checkDuplicates(slotData);
    setNameErrors(errors);
  }, [slotData, checkDuplicates]);

  const handleNameChange = (linkId: number, value: string) => {
    setSlotData(prev =>
      prev.map(slot => (slot.linkId === linkId ? { ...slot, currentName: value } : slot)),
    );
  };

  const handleSave = () => {
    slotData.forEach(slot => {
      const trimmedName = slot.currentName.trim();
      updateSlotLabel(nodeId, slot.linkId, trimmedName);
    });
    onClose();
  };

  const hasErrors = Object.values(nameErrors).some(error => error);

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>Edit Slot Names</span>
          <button className={styles.closeBtn} onClick={onClose}>
            ×
          </button>
        </div>

        <div className={styles.body}>
          <div className={styles.slotList}>
            {slotData.map(slot => (
              <div key={slot.linkId} className={styles.slotItem}>
                <div className={styles.slotInfo}>
                  <span className={styles.slotId}>Link {slot.linkId}</span>
                  <span className={styles.slotType}>{slot.type}</span>
                </div>
                <div className={styles.field}>
                  <input
                    className={`${styles.input} ${nameErrors[slot.linkId] ? styles.inputError : ''}`}
                    type='text'
                    value={slot.currentName}
                    onChange={e => handleNameChange(slot.linkId, e.target.value)}
                    placeholder='Enter slot name...'
                  />
                  {nameErrors[slot.linkId] && (
                    <div className={styles.errorMsg}>
                      ⚠ Name already exists! Please use a unique name.
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
