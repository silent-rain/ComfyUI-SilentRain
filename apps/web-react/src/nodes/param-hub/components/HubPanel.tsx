import React, { useState, useEffect, useRef } from 'react';
import { EditSlotModal } from './EditSlotModal';
import styles from './HubPanel.module.scss';
import type { NodeId } from '@/types/comfy';

interface HubPanelProps {
  nodeId: NodeId;
}

/**
 * ParamHub 节点的 React UI 面板。
 *
 * 通过 DOM widget 挂载到 LiteGraph 节点内部，
 * 可使用完整的 React 特性：Hooks、Zustand Store、事件处理等。
 */
export const HubPanel: React.FC<HubPanelProps> = ({ nodeId }) => {
  // 编辑弹窗状态
  const [showEditModal, setShowEditModal] = useState(false);
  // 引用容器元素
  const containerRef = useRef<HTMLDivElement>(null);

  const handleEditClick = () => {
    setShowEditModal(true);
  };

  const handleCloseModal = () => {
    setShowEditModal(false);
  };

  // 组件挂载后，通知 LiteGraph 调整节点大小
  useEffect(() => {
    // 延迟执行，确保 DOM 已渲染
    const timer = setTimeout(() => {
      // 查找父级 LiteGraph 节点并触发大小调整
      const nodeElement = containerRef.current?.closest('.comfy-node, .litegraph-node');
      if (nodeElement) {
        // 触发 resize 事件
        const event = new Event('resize', { bubbles: true });
        nodeElement.dispatchEvent(event);
      }

      // 尝试直接调用 LiteGraph 的节点大小调整
      // 通过全局 ComfyUI API
      if (window.app?.rootGraph) {
        window.app.rootGraph.setDirtyCanvas(true, true);
      }
    }, 50);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={styles.srHubPanel} ref={containerRef}>
      <div className={styles.srHubHeader}>
        <button className={styles.srEditAllBtn} onClick={handleEditClick} title='Edit All Slots'>
          Edit Slots
        </button>
      </div>

      {/* 编辑弹窗 */}
      {showEditModal && <EditSlotModal nodeId={nodeId} onClose={handleCloseModal} />}
    </div>
  );
};
