interface ThumbnailCardProps {
  src: string;
  alt: string;
  onClick: (src: string, caption: string) => void;
  delay?: number;
}

export const ThumbnailCard = ({ src, alt, onClick, delay = 0 }: ThumbnailCardProps) => {
  return (
    <div 
      className="group bg-thumbnail-bg rounded-xl p-4 border border-border shadow-sm hover:shadow-hover hover:-translate-y-2 transition-all duration-300 cursor-pointer animate-fade-in-scale"
      style={{ animationDelay: `${delay}ms` }}
      onClick={() => onClick(src, alt)}
    >
      <div className="relative overflow-hidden rounded-lg bg-white mb-3 aspect-video">
        <img
          src={src}
          alt={alt}
          className="w-full h-full object-contain transition-transform duration-300 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
      </div>
      <div className="text-sm text-muted-foreground text-center break-words group-hover:text-foreground transition-colors duration-300">
        {alt}
      </div>
    </div>
  );
};
