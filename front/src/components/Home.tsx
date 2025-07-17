"use client";

import { useRef, useCallback, useEffect, useState } from "react";
import useSWRInfinite from "swr/infinite";
import VideoCard from "../components/VideoCard";
import { motion } from "framer-motion";
import { Heart, MessageCircle, Share2 } from "lucide-react";
import { videoData } from "src/api/tmp_data";
import { Video } from "src/types";
import Image from "next/image";

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function Home() {
	const [currentVideoIndex, setCurrentVideoIndex] = useState<string | null>(
		null
	);
	const containerRef = useRef(null);
	const videoRefs = useRef<HTMLDivElement[]>([]);
	const actionButtonRefs = useRef({});

	const loadMoreRef = useRef<HTMLDivElement>(null);

	const getKey = (pageIndex: number, previousPageData: Video[]) => {
		if (previousPageData && previousPageData.length === 0) return null;
		return `/api/new_video`;
	};

	const { data, size, setSize, isValidating } = useSWRInfinite<Video[]>(
		getKey,
		fetcher,
		{ revalidateOnFocus: false }
	);

	// const videos = data ? ([] as Video[]).concat(...data) : [];
	const videos = videoData;

	const onIntersect: IntersectionObserverCallback = useCallback(
		([entry]) => {
			if (entry.isIntersecting && !isValidating) {
				setSize(size + 1);
			}
		},
		[isValidating, setSize, size]
	);

	useEffect(() => {
		const observer = new IntersectionObserver(onIntersect, {
			rootMargin: "200px",
		});
		if (loadMoreRef.current) observer.observe(loadMoreRef.current);
		return () => observer.disconnect();
	}, [onIntersect]);

	return (
		<div className="h-full w-full max-w-md mx-auto bg-black relative overflow-hidden">
			{/* Main Container */}
			<div
				ref={containerRef}
				className="h-full w-full overflow-y-scroll snap-y snap-mandatory scroll-smooth"
				style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
			>
				<div className="text-white text-lg font-bold animate-on-mount absolute z-10 p-4 flex flex-row gap-2">
					<Image
						alt="logo"
						src="/logo.png"
						width={40}
						height={40}
						className="object-contain pt-0.5"
					/>
					<div>•</div> For You
				</div>
				<style jsx>{`
					div::-webkit-scrollbar {
						display: none;
					}
				`}</style>

				{videos.map((video) => (
					<div
						key={video.id}
						className="snap-start h-full w-full flex-shrink-0"
						// ref={(el) =>
						// 	videoRefs !== null
						// 		? (videoRefs.current[index] = el)
						// 		: null
						// }
					>
						{
							<VideoCard
								video={video}
								index={video.id}
								setCurrentVideoIndex={setCurrentVideoIndex}
							/>
						}
					</div>
				))}
			</div>
		</div>
	);
}
