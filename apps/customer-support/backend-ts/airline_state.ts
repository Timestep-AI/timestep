/**
 * AirlineStateManager manages per-thread airline customer state.
 */

function nowIso(): string {
  return new Date().toISOString();
}

export interface TimelineEntry {
  timestamp: string;
  kind: string;
  entry: string;
}

export class FlightSegment {
  flightNumber: string;
  date: string;
  origin: string;
  destination: string;
  departureTime: string;
  arrivalTime: string;
  seat: string;
  status: string;

  constructor(
    flightNumber: string,
    date: string,
    origin: string,
    destination: string,
    departureTime: string,
    arrivalTime: string,
    seat: string,
    status: string = "Scheduled"
  ) {
    this.flightNumber = flightNumber;
    this.date = date;
    this.origin = origin;
    this.destination = destination;
    this.departureTime = departureTime;
    this.arrivalTime = arrivalTime;
    this.seat = seat;
    this.status = status;
  }

  cancel(): void {
    this.status = "Cancelled";
  }

  changeSeat(newSeat: string): void {
    this.seat = newSeat;
  }

  toDict(): Record<string, unknown> {
    return {
      flight_number: this.flightNumber,
      date: this.date,
      origin: this.origin,
      destination: this.destination,
      departure_time: this.departureTime,
      arrival_time: this.arrivalTime,
      seat: this.seat,
      status: this.status,
    };
  }
}

export class CustomerProfile {
  customerId: string;
  name: string;
  loyaltyStatus: string;
  loyaltyId: string;
  email: string;
  phone: string;
  tierBenefits: string[];
  segments: FlightSegment[];
  bagsChecked: number;
  mealPreference: string | null;
  specialAssistance: string | null;
  timeline: TimelineEntry[];

  constructor(
    customerId: string,
    name: string,
    loyaltyStatus: string,
    loyaltyId: string,
    email: string,
    phone: string,
    tierBenefits: string[],
    segments: FlightSegment[],
    bagsChecked: number = 0,
    mealPreference: string | null = null,
    specialAssistance: string | null = null,
    timeline: TimelineEntry[] = []
  ) {
    this.customerId = customerId;
    this.name = name;
    this.loyaltyStatus = loyaltyStatus;
    this.loyaltyId = loyaltyId;
    this.email = email;
    this.phone = phone;
    this.tierBenefits = tierBenefits;
    this.segments = segments;
    this.bagsChecked = bagsChecked;
    this.mealPreference = mealPreference;
    this.specialAssistance = specialAssistance;
    this.timeline = timeline;
  }

  log(entry: string, kind: string = "info"): void {
    this.timeline.unshift({ timestamp: nowIso(), kind, entry });
  }

  toDict(): Record<string, unknown> {
    return {
      customer_id: this.customerId,
      name: this.name,
      loyalty_status: this.loyaltyStatus,
      loyalty_id: this.loyaltyId,
      email: this.email,
      phone: this.phone,
      tier_benefits: this.tierBenefits,
      segments: this.segments.map((s) => s.toDict()),
      bags_checked: this.bagsChecked,
      meal_preference: this.mealPreference,
      special_assistance: this.specialAssistance,
      timeline: this.timeline,
    };
  }
}

export class AirlineStateManager {
  private states: Map<string, CustomerProfile> = new Map();

  private createDefaultState(): CustomerProfile {
    const segments = [
      new FlightSegment(
        "OA476",
        "2025-10-02",
        "SFO",
        "JFK",
        "08:05",
        "16:35",
        "14A"
      ),
      new FlightSegment(
        "OA477",
        "2025-10-10",
        "JFK",
        "SFO",
        "18:50",
        "22:15",
        "15C"
      ),
    ];
    const profile = new CustomerProfile(
      "cus_98421",
      "Jordan Miles",
      "Aviator Platinum",
      "APL-204981",
      "jordan.miles@example.com",
      "+1 (415) 555-9214",
      [
        "Complimentary upgrades when available",
        "Unlimited lounge access",
        "Priority boarding group 1",
      ],
      segments
    );
    profile.log("Itinerary imported from confirmation LL0EZ6.", "system");
    return profile;
  }

  getProfile(threadId: string): CustomerProfile {
    if (!this.states.has(threadId)) {
      this.states.set(threadId, this.createDefaultState());
    }
    return this.states.get(threadId)!;
  }

  changeSeat(threadId: string, flightNumber: string, seat: string): string {
    const profile = this.getProfile(threadId);
    if (!this.isValidSeat(seat)) {
      throw new Error("Seat must be a row number followed by a letter, for example 12C.");
    }

    const segment = this.findSegment(profile, flightNumber);
    if (segment === null) {
      throw new Error(`Flight ${flightNumber} is not on the customer's itinerary.`);
    }

    const previous = segment.seat;
    segment.changeSeat(seat.toUpperCase());
    profile.log(
      `Seat changed on ${segment.flightNumber} from ${previous} to ${segment.seat}.`,
      "success"
    );
    return `Seat updated to ${segment.seat} on flight ${segment.flightNumber}.`;
  }

  cancelTrip(threadId: string): string {
    const profile = this.getProfile(threadId);
    for (const segment of profile.segments) {
      segment.cancel();
    }
    profile.log("Trip cancelled at customer request.", "warning");
    return "The reservation has been cancelled. Refund processing will begin immediately.";
  }

  addBag(threadId: string): string {
    const profile = this.getProfile(threadId);
    profile.bagsChecked += 1;
    profile.log(`Added checked bag. Total bags now ${profile.bagsChecked}.`, "info");
    return `Checked bag added. You now have ${profile.bagsChecked} bag(s) checked.`;
  }

  setMeal(threadId: string, meal: string): string {
    const profile = this.getProfile(threadId);
    profile.mealPreference = meal;
    profile.log(`Meal preference updated to ${meal}.`, "info");
    return `We'll note ${meal} as the meal preference.`;
  }

  requestAssistance(threadId: string, note: string): string {
    const profile = this.getProfile(threadId);
    profile.specialAssistance = note;
    profile.log(`Special assistance noted: ${note}.`, "info");
    return "Assistance request recorded. Airport staff will be notified.";
  }

  toDict(threadId: string): Record<string, unknown> {
    return this.getProfile(threadId).toDict();
  }

  private isValidSeat(seat: string): boolean {
    const trimmed = seat.trim().toUpperCase();
    if (trimmed.length < 2) {
      return false;
    }
    const row = trimmed.slice(0, -1);
    const letter = trimmed.slice(-1);
    return /^\d+$/.test(row) && /^[A-Z]$/.test(letter);
  }

  private findSegment(profile: CustomerProfile, flightNumber: string): FlightSegment | null {
    const normalized = flightNumber.toUpperCase().trim();
    for (const segment of profile.segments) {
      if (segment.flightNumber.toUpperCase() === normalized) {
        return segment;
      }
    }
    return null;
  }
}

