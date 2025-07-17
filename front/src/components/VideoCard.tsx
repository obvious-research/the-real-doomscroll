"use client";

import { useAnimation, motion } from "framer-motion";
import { useRef, useEffect, useState, Dispatch, SetStateAction } from "react";
import { Video } from "src/types";
import {
	Heart,
	MessageCircle,
	Share2,
	Play,
	Volume2,
	VolumeX,
	MoreHorizontal,
	Plus,
} from "lucide-react";
import { formatNumber } from "src/utils";
import Image from "next/image";

type VideoCardProps = {
	video: Video;
	index: string;
	setCurrentVideoIndex: Dispatch<SetStateAction<string | null>>;
};

export default function VideoCard({
	video,
	index,
	setCurrentVideoIndex,
}: VideoCardProps) {
	const controls = useAnimation();
	const containerRef = useRef<HTMLDivElement>(null);
	const videoRef = useRef<HTMLVideoElement>(null);
	const [videoLoaded, setVideoLoaded] = useState(false);
	const [isPlaying, setIsPlaying] = useState(false);
	const [isMuted, setIsMuted] = useState(false);

	useEffect(() => {
		console.log("isVideoLoaded", videoLoaded);
	}, [videoLoaded]);

	useEffect(() => {
		const observer = new IntersectionObserver(
			([entry]) => {
				if (entry.isIntersecting) {
					controls.start("visible");
					setCurrentVideoIndex(index);
					playVideo(true);
				} else {
					controls.start("hidden");
					pauseVideo();
				}
			},
			{ threshold: 0.5 }
		);
		if (containerRef.current) observer.observe(containerRef.current);
		return () => observer.disconnect();
	}, [controls]);

	const playVideo = (from_beginning?: boolean) => {
		if (videoRef.current) {
			if (from_beginning === true) videoRef.current.currentTime = 0;
			const playPromise = videoRef.current.play();

			if (playPromise !== undefined) {
				playPromise
					.then(() => {
						setIsPlaying(true);
						setVideoLoaded(true);
					})
					.catch((error) => {
						console.log("Auto-play failed:", error);
						setIsPlaying(false);
					});
			}
		}
	};
	const pauseVideo = () => {
		if (videoRef.current) {
			videoRef.current.pause();
			setIsPlaying(false);
		}
	};

	return (
		<motion.div
			ref={containerRef}
			initial="hidden"
			animate={controls}
			variants={{ hidden: { opacity: 0.5 }, visible: { opacity: 1 } }}
			transition={{ duration: 0.3 }}
			className="sm:h-screen h-dvh  w-full snap-start flex items-center justify-center bg-black relative"
			onClick={(e) => {
				e.preventDefault();
				isPlaying ? pauseVideo() : playVideo();
			}}
		>
			<div
				className="absolute inset-0 flex items-center justify-center cursor-pointer"
				onClick={() => (isPlaying ? pauseVideo() : playVideo())}
			>
				{!isPlaying && (
					<div className="play-button rounded-full">
						<Play size={48} className="text-white ml-2" />
					</div>
				)}
			</div>

			<div className="absolute flex right-0 top-0 p-4">
				<motion.button
					onClick={(e) => {
						e.preventDefault();
						setIsMuted(!isMuted);
					}}
					className="text-white mute-button animate-on-mount pe-2 z-10"
					whileHover={{ scale: 1.1 }}
					whileTap={{ scale: 0.95 }}
					transition={{ duration: 0.2, ease: "easeOut" }}
				>
					{isMuted ? <VolumeX size={24} /> : <Volume2 size={24} />}
				</motion.button>
			</div>
			{/* Top Controls */}
			<div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/50 to-transparent justify-end flex">
				<div className="flex flex-col items-center justify-between">
					<div className="flex flex-col items-center space-y-6 p-3">
						<div className="action-button relative w-12 h-12">
							<Image
								fill={true}
								src={"/avatar.png"}
								alt={"avatar"}
								className="w-12 h-12 rounded-full border-2 border-white"
							/>
							<div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 bg-red-500 rounded-full p-1">
								<Plus size={12} className="text-white" />
							</div>
						</div>
						<button className="action-button flex flex-col items-center group">
							<div
								className={` p-2 rounded-full transition-all text-white `}
							>
								<Heart size={24} />
							</div>
							<span className="text-white text-xs mt-1 font-medium">
								{formatNumber(video.likes_count)}
							</span>
						</button>

						<button className="action-button flex flex-col items-center group">
							<div className="p-2 rounded-full bg-opacity-50 text-white hover:bg-blue-500 transition-all">
								<MessageCircle size={24} />
							</div>
							<span className="text-white text-xs mt-1 font-medium">
								{formatNumber(video.comments_count)}
							</span>
						</button>

						<button className="action-button flex flex-col items-center group">
							<div className="p-2 rounded-full bg-opacity-50 text-white hover:bg-green-500 transition-all">
								<Share2 size={24} />
							</div>
							<span className="text-white text-xs mt-1 font-medium">
								{formatNumber(video.share_count)}
							</span>
						</button>

						<button className="action-button flex flex-col items-center group">
							<div className="p-2 rounded-full bg-opacity-50 text-white hover:bg-gray-600 transition-all">
								<MoreHorizontal size={24} />
							</div>
						</button>
					</div>
				</div>
			</div>

			<video
				ref={videoRef}
				src={video.src}
				title={video.title}
				controls={false}
				autoPlay
				playsInline
				muted={true}
				loop
				className={`w-full h-full object-cover ${
					videoLoaded === true ? "" : "opacity-0"
				}`}
				onLoadedData={() => setVideoLoaded(true)}
			/>

			<div className="absolute bottom-0 bg-gradient-to-b text-white text-start w-full p-4">
				<h2 className="text-lg font-semibold">{video.title}</h2>
				<h2 className="text-lg">{video.description}</h2>
				<h2 className="text-lg font-semibold">{video.hashtags}</h2>
			</div>
		</motion.div>
	);
}
