from googleapiclient.discovery import build
import sys

def get_youtube_comments(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
            comments.append(f"{author}: {comment}")

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_youtube.py VIDEO_ID")
        sys.exit(1)

    video_id = sys.argv[1].strip()
    api_key = "AIzaSyBu2MurIA1F_yiMlG-WzGDGW2lnani5h0Q"

    print("Collecting YouTube comments...")
    comments = get_youtube_comments(api_key, video_id)

    if comments:
        print(f"✅ Collected {len(comments)} comments!")
        with open("youtube_comments.txt", "w", encoding="utf-8") as f:
            for c in comments:
                f.write(c + "\n---\n")
        print("💾 Saved to youtube_comments.txt")
    else:
        print("No comments found.")
