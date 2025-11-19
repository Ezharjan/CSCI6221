import { ThumbnailCard } from "./ThumbnailCard";

interface FileSectionProps {
  fileName: string;
  charts: { src: string; alt: string }[];
  onImageClick: (src: string, caption: string) => void;
  delay?: number;
}

export const FileSection = ({ fileName, charts, onImageClick, delay = 0 }: FileSectionProps) => {
  return (
    <section 
      className="bg-section-bg border border-border rounded-2xl p-6 mb-6 shadow-md hover:shadow-hover transition-all duration-300 hover:-translate-y-1 animate-fade-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      <h2 className="text-2xl font-semibold text-foreground mb-4 pb-3 border-b border-border">
        {fileName}
      </h2>
      
      {charts.length === 0 ? (
        <p className="text-muted-foreground text-center py-8 italic">
          No charts created for this file.
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {charts.map((chart, index) => (
            <ThumbnailCard
              key={chart.src}
              src={chart.src}
              alt={chart.alt}
              onClick={onImageClick}
              delay={delay + (index * 100)}
            />
          ))}
        </div>
      )}
    </section>
  );
};
