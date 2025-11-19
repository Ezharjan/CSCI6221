import { useState } from "react";
import { DashboardHeader } from "@/components/DashboardHeader";
import { FileSection } from "@/components/FileSection";
import { ImageModal } from "@/components/ImageModal";

const Index = () => {
  const [modalOpen, setModalOpen] = useState(false);
  const [modalSrc, setModalSrc] = useState("");
  const [modalCaption, setModalCaption] = useState("");

  const handleImageClick = (src: string, caption: string) => {
    setModalSrc(src);
    setModalCaption(caption);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
    setTimeout(() => {
      setModalSrc("");
      setModalCaption("");
    }, 300);
  };

  const dashboardData = {
    generatedDate: "2025-10-22T20:06:31.102685",
    files: [
      {
        fileName: "full_benchmark_comparison.json",
        charts: []
      },
      {
        fileName: "full_benchmark_scalability.json",
        charts: [
          { src: "scalability_analysis.png", alt: "scalability_analysis.png" }
        ]
      },
      {
        fileName: "full_benchmark_stress.json",
        charts: [
          { src: "stress_test__highload_chart.png", alt: "stress_test__highload_chart.png" },
          { src: "stress_test__longduration_chart.png", alt: "stress_test__longduration_chart.png" },
          { src: "stress_test__megaload_chart.png", alt: "stress_test__megaload_chart.png" }
        ]
      },
      {
        fileName: "scalability_evaluation.json",
        charts: [
          { src: "scalability_analysis.png", alt: "scalability_analysis.png" }
        ]
      }
    ]
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <DashboardHeader generatedDate={dashboardData.generatedDate} />
        
        {dashboardData.files.map((file, index) => (
          <FileSection
            key={file.fileName}
            fileName={file.fileName}
            charts={file.charts}
            onImageClick={handleImageClick}
            delay={index * 100}
          />
        ))}
      </div>

      <ImageModal
        isOpen={modalOpen}
        src={modalSrc}
        caption={modalCaption}
        onClose={handleCloseModal}
      />
    </div>
  );
};

export default Index;
