package vega

import (
	"context"
	"testing"
	"time"
)

// mockLLM is a simple mock for testing
type mockLLM struct {
	response string
	err      error
}

func (m *mockLLM) Generate(ctx context.Context, messages []Message, tools []ToolSchema) (*LLMResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &LLMResponse{
		Content:      m.response,
		InputTokens:  10,
		OutputTokens: 5,
		CostUSD:      0.001,
	}, nil
}

func (m *mockLLM) GenerateStream(ctx context.Context, messages []Message, tools []ToolSchema) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, 1)
	go func() {
		ch <- StreamEvent{Delta: m.response}
		close(ch)
	}()
	return ch, nil
}

func TestNewOrchestrator(t *testing.T) {
	o := NewOrchestrator()
	if o == nil {
		t.Fatal("NewOrchestrator() returned nil")
	}

	if o.maxProcesses != 100 {
		t.Errorf("Default maxProcesses = %d, want 100", o.maxProcesses)
	}

	if len(o.processes) != 0 {
		t.Errorf("New orchestrator should have 0 processes, got %d", len(o.processes))
	}
}

func TestWithMaxProcesses(t *testing.T) {
	o := NewOrchestrator(WithMaxProcesses(50))
	if o.maxProcesses != 50 {
		t.Errorf("maxProcesses = %d, want 50", o.maxProcesses)
	}
}

func TestWithLLM(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))
	if o.defaultLLM != llm {
		t.Error("defaultLLM was not set correctly")
	}
}

func TestSpawn(t *testing.T) {
	llm := &mockLLM{response: "Hello!"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{
		Name:   "test-agent",
		Model:  "test-model",
		System: StaticPrompt("You are a test agent."),
	}

	proc, err := o.Spawn(agent)
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	if proc == nil {
		t.Fatal("Spawn() returned nil process")
	}

	if proc.ID == "" {
		t.Error("Process should have an ID")
	}

	if proc.Agent.Name != "test-agent" {
		t.Errorf("Process.Agent.Name = %q, want %q", proc.Agent.Name, "test-agent")
	}

	if proc.Status() != StatusRunning {
		t.Errorf("Process.Status() = %q, want %q", proc.Status(), StatusRunning)
	}
}

func TestSpawnWithTask(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	proc, err := o.Spawn(agent, WithTask("Build a REST API"))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	if proc.Task != "Build a REST API" {
		t.Errorf("Process.Task = %q, want %q", proc.Task, "Build a REST API")
	}
}

func TestSpawnWithWorkDir(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	proc, err := o.Spawn(agent, WithWorkDir("/tmp/test"))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	if proc.WorkDir != "/tmp/test" {
		t.Errorf("Process.WorkDir = %q, want %q", proc.WorkDir, "/tmp/test")
	}
}

func TestSpawnMaxProcesses(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm), WithMaxProcesses(2))

	agent := Agent{Name: "test"}

	// Spawn 2 processes (should succeed)
	_, err := o.Spawn(agent)
	if err != nil {
		t.Fatalf("First Spawn() returned error: %v", err)
	}

	_, err = o.Spawn(agent)
	if err != nil {
		t.Fatalf("Second Spawn() returned error: %v", err)
	}

	// Third should fail
	_, err = o.Spawn(agent)
	if err != ErrMaxProcessesReached {
		t.Errorf("Third Spawn() error = %v, want ErrMaxProcessesReached", err)
	}
}

func TestSpawnWithoutLLM(t *testing.T) {
	o := NewOrchestrator() // No LLM configured

	agent := Agent{Name: "test"}
	_, err := o.Spawn(agent)
	if err == nil {
		t.Error("Spawn() without LLM should return error")
	}
}

func TestGet(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	proc, _ := o.Spawn(agent)

	// Get existing process
	got := o.Get(proc.ID)
	if got != proc {
		t.Error("Get() did not return the spawned process")
	}

	// Get non-existent process
	got = o.Get("nonexistent")
	if got != nil {
		t.Error("Get() for non-existent ID should return nil")
	}
}

func TestList(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	// Empty list
	procs := o.List()
	if len(procs) != 0 {
		t.Errorf("List() on empty orchestrator = %d processes, want 0", len(procs))
	}

	// Spawn some processes
	agent := Agent{Name: "test"}
	o.Spawn(agent)
	o.Spawn(agent)

	procs = o.List()
	if len(procs) != 2 {
		t.Errorf("List() = %d processes, want 2", len(procs))
	}
}

func TestKill(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	proc, _ := o.Spawn(agent)
	id := proc.ID

	// Kill existing process
	err := o.Kill(id)
	if err != nil {
		t.Errorf("Kill() returned error: %v", err)
	}

	// Process should be removed
	if o.Get(id) != nil {
		t.Error("Process should be removed after Kill()")
	}

	// Kill non-existent process
	err = o.Kill("nonexistent")
	if err != ErrProcessNotFound {
		t.Errorf("Kill() for non-existent ID = %v, want ErrProcessNotFound", err)
	}
}

func TestShutdown(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	o.Spawn(agent)
	o.Spawn(agent)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := o.Shutdown(ctx)
	if err != nil {
		t.Errorf("Shutdown() returned error: %v", err)
	}
}

func TestShutdownTimeout(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	// Create a context that's already cancelled
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := o.Shutdown(ctx)
	if err != context.Canceled {
		t.Errorf("Shutdown() with cancelled context = %v, want context.Canceled", err)
	}
}

func TestWithRateLimits(t *testing.T) {
	limits := map[string]RateLimitConfig{
		"claude-3": {
			RequestsPerMinute: 60,
			TokensPerMinute:   100000,
			Strategy:          RateLimitQueue,
		},
	}

	o := NewOrchestrator(WithRateLimits(limits))
	if len(o.rateLimits) != 1 {
		t.Errorf("rateLimits count = %d, want 1", len(o.rateLimits))
	}
}

func TestRateLimitStrategy(t *testing.T) {
	tests := []struct {
		strategy RateLimitStrategy
		want     RateLimitStrategy
	}{
		{RateLimitQueue, 0},
		{RateLimitReject, 1},
		{RateLimitBackpressure, 2},
	}

	for _, tt := range tests {
		if tt.strategy != tt.want {
			t.Errorf("RateLimitStrategy = %d, want %d", tt.strategy, tt.want)
		}
	}
}

func TestWithSupervision(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	sup := Supervision{
		Strategy:    Restart,
		MaxRestarts: 3,
	}

	agent := Agent{Name: "test"}
	proc, err := o.Spawn(agent, WithSupervision(sup))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	if proc.Supervision == nil {
		t.Error("Process.Supervision should not be nil")
	}

	if proc.Supervision.MaxRestarts != 3 {
		t.Errorf("Process.Supervision.MaxRestarts = %d, want 3", proc.Supervision.MaxRestarts)
	}
}

func TestWithTimeout(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	proc, err := o.Spawn(agent, WithTimeout(5*time.Second))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	// The context should have a deadline
	deadline, ok := proc.ctx.Deadline()
	if !ok {
		t.Error("Process context should have a deadline")
	}

	// Deadline should be roughly 5 seconds from now
	remaining := time.Until(deadline)
	if remaining < 4*time.Second || remaining > 6*time.Second {
		t.Errorf("Context deadline remaining = %v, want ~5s", remaining)
	}
}

func TestWithProcessContext(t *testing.T) {
	llm := &mockLLM{response: "test"}
	o := NewOrchestrator(WithLLM(llm))

	parentCtx, parentCancel := context.WithCancel(context.Background())
	defer parentCancel()

	agent := Agent{Name: "test"}
	proc, err := o.Spawn(agent, WithProcessContext(parentCtx))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	// Cancel parent should affect child
	parentCancel()

	select {
	case <-proc.ctx.Done():
		// Expected
	case <-time.After(100 * time.Millisecond):
		t.Error("Process context should be cancelled when parent is cancelled")
	}
}

// --- Named Process Tests ---

func TestRegister(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}
	proc, _ := o.Spawn(agent)

	err := o.Register("worker-1", proc)
	if err != nil {
		t.Fatalf("Register() returned error: %v", err)
	}

	if proc.Name() != "worker-1" {
		t.Errorf("proc.Name() = %q, want %q", proc.Name(), "worker-1")
	}
}

func TestRegisterDuplicate(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}
	proc1, _ := o.Spawn(agent)
	proc2, _ := o.Spawn(agent)

	o.Register("worker", proc1)
	err := o.Register("worker", proc2)

	if err == nil {
		t.Error("Register() should return error for duplicate name")
	}
}

func TestRegisterSameProcessTwice(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}
	proc, _ := o.Spawn(agent)

	o.Register("worker", proc)
	err := o.Register("worker", proc) // Same process, same name

	if err != nil {
		t.Errorf("Re-registering same process with same name should be idempotent, got: %v", err)
	}
}

func TestGetByName(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}
	proc, _ := o.Spawn(agent)

	o.Register("worker", proc)
	found := o.GetByName("worker")

	if found != proc {
		t.Error("GetByName() should return the registered process")
	}
}

func TestGetByNameNotFound(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))

	found := o.GetByName("nonexistent")
	if found != nil {
		t.Error("GetByName() should return nil for unknown name")
	}
}

func TestUnregister(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}
	proc, _ := o.Spawn(agent)

	o.Register("worker", proc)
	o.Unregister("worker")

	if o.GetByName("worker") != nil {
		t.Error("After Unregister(), name should not be found")
	}
	if proc.Name() != "" {
		t.Error("After Unregister(), process name should be empty")
	}
}

func TestNameUnregisteredOnComplete(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}
	proc, _ := o.Spawn(agent)

	o.Register("worker", proc)
	proc.Complete("done")

	time.Sleep(10 * time.Millisecond)

	if o.GetByName("worker") != nil {
		t.Error("Name should be unregistered when process completes")
	}
}

func TestNameUnregisteredOnFail(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}
	proc, _ := o.Spawn(agent)

	o.Register("worker", proc)
	proc.Fail(ErrTimeout)

	time.Sleep(10 * time.Millisecond)

	if o.GetByName("worker") != nil {
		t.Error("Name should be unregistered when process fails")
	}
}

// --- Supervision Tree Tests ---

func TestSupervisorStrategy(t *testing.T) {
	tests := []struct {
		strategy SupervisorStrategy
		want     string
	}{
		{OneForOne, "one_for_one"},
		{OneForAll, "one_for_all"},
		{RestForOne, "rest_for_one"},
	}

	for _, tt := range tests {
		if tt.strategy.String() != tt.want {
			t.Errorf("SupervisorStrategy.String() = %q, want %q", tt.strategy.String(), tt.want)
		}
	}
}

func TestChildRestart(t *testing.T) {
	tests := []struct {
		restart ChildRestart
		want    string
	}{
		{Permanent, "permanent"},
		{Transient, "transient"},
		{Temporary, "temporary"},
	}

	for _, tt := range tests {
		if tt.restart.String() != tt.want {
			t.Errorf("ChildRestart.String() = %q, want %q", tt.restart.String(), tt.want)
		}
	}
}

func TestNewSupervisor(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	spec := SupervisorSpec{
		Strategy:    OneForOne,
		MaxRestarts: 5,
		Window:      time.Minute,
		Children: []ChildSpec{
			{Name: "worker1", Agent: Agent{Name: "Worker"}, Restart: Permanent},
		},
	}

	sup := o.NewSupervisor(spec)
	if sup == nil {
		t.Fatal("NewSupervisor() returned nil")
	}
	if sup.spec.Strategy != OneForOne {
		t.Errorf("Supervisor strategy = %v, want %v", sup.spec.Strategy, OneForOne)
	}
}

func TestSupervisorStart(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	spec := SupervisorSpec{
		Strategy: OneForOne,
		Children: []ChildSpec{
			{Name: "worker1", Agent: Agent{Name: "Worker1"}, Restart: Permanent},
			{Name: "worker2", Agent: Agent{Name: "Worker2"}, Restart: Permanent},
		},
	}

	sup := o.NewSupervisor(spec)
	err := sup.Start()
	if err != nil {
		t.Fatalf("Supervisor.Start() returned error: %v", err)
	}
	defer sup.Stop()

	children := sup.Children()
	if len(children) != 2 {
		t.Errorf("Supervisor has %d children, want 2", len(children))
	}

	// Check names are registered
	if o.GetByName("worker1") == nil {
		t.Error("worker1 should be registered")
	}
	if o.GetByName("worker2") == nil {
		t.Error("worker2 should be registered")
	}
}

func TestSupervisorStop(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	spec := SupervisorSpec{
		Strategy: OneForOne,
		Children: []ChildSpec{
			{Name: "worker1", Agent: Agent{Name: "Worker1"}, Restart: Permanent},
		},
	}

	sup := o.NewSupervisor(spec)
	sup.Start()
	sup.Stop()

	time.Sleep(10 * time.Millisecond)

	// Names should be unregistered
	if o.GetByName("worker1") != nil {
		t.Error("worker1 should be unregistered after Stop()")
	}
}

func TestSupervisorChildren(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	spec := SupervisorSpec{
		Strategy: OneForOne,
		Children: []ChildSpec{
			{Agent: Agent{Name: "Worker1"}, Restart: Permanent},
			{Agent: Agent{Name: "Worker2"}, Restart: Permanent},
			{Agent: Agent{Name: "Worker3"}, Restart: Permanent},
		},
	}

	sup := o.NewSupervisor(spec)
	sup.Start()
	defer sup.Stop()

	children := sup.Children()
	if len(children) != 3 {
		t.Errorf("len(Children()) = %d, want 3", len(children))
	}

	for i, child := range children {
		if child.Status() != StatusRunning {
			t.Errorf("Child %d status = %v, want %v", i, child.Status(), StatusRunning)
		}
	}
}

// --- Automatic Restart Tests ---

func TestSpawnSupervised(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}

	proc, err := o.SpawnSupervised(agent, Permanent)
	if err != nil {
		t.Fatalf("SpawnSupervised() returned error: %v", err)
	}

	if proc.Status() != StatusRunning {
		t.Errorf("Process status = %v, want %v", proc.Status(), StatusRunning)
	}

	// Agent should be registered for respawning
	_, ok := o.GetAgent("TestAgent")
	if !ok {
		t.Error("Agent should be registered for respawning")
	}
}

func TestRegisterAgent(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "MyAgent"}

	o.RegisterAgent(agent)

	found, ok := o.GetAgent("MyAgent")
	if !ok {
		t.Error("GetAgent() should find registered agent")
	}
	if found.Name != "MyAgent" {
		t.Errorf("Agent name = %q, want %q", found.Name, "MyAgent")
	}
}

func TestAutomaticRestartPermanent(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}

	proc, _ := o.SpawnSupervised(agent, Permanent, WithTask("test task"))
	originalID := proc.ID

	// Fail the process
	proc.Fail(ErrTimeout)

	// Wait for restart
	time.Sleep(100 * time.Millisecond)

	// Check that a new process was spawned
	procs := o.List()
	found := false
	for _, p := range procs {
		if p.ID != originalID && p.Status() == StatusRunning {
			found = true
			break
		}
	}

	if !found {
		t.Error("Permanent process should be automatically restarted")
	}
}

func TestAutomaticRestartTemporary(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}

	proc, _ := o.SpawnSupervised(agent, Temporary)
	originalID := proc.ID

	// Fail the process
	proc.Fail(ErrTimeout)

	// Wait a bit
	time.Sleep(100 * time.Millisecond)

	// Check that NO new process was spawned
	procs := o.List()
	for _, p := range procs {
		if p.ID != originalID && p.Status() == StatusRunning {
			t.Error("Temporary process should NOT be automatically restarted")
		}
	}
}

func TestAutomaticRestartWithSupervision(t *testing.T) {
	o := NewOrchestrator(WithLLM(&mockLLM{}))
	agent := Agent{Name: "TestAgent"}

	restartCount := 0
	sup := Supervision{
		Strategy:    Restart,
		MaxRestarts: 2,
		Window:      time.Minute,
		OnRestart: func(p *Process, attempt int) {
			restartCount = attempt
		},
	}

	proc, _ := o.SpawnSupervised(agent, Permanent, WithSupervision(sup))

	// Fail the process
	proc.Fail(ErrTimeout)

	// Wait for restart
	time.Sleep(100 * time.Millisecond)

	if restartCount == 0 {
		t.Error("OnRestart callback should have been called")
	}
}

// --- WithMessages Tests ---

func TestWithMessages(t *testing.T) {
	llm := &mockLLM{response: "I remember that!"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{
		Name:   "test-agent",
		System: StaticPrompt("You are a helpful assistant."),
	}

	// Spawn with existing conversation history
	history := []Message{
		{Role: RoleUser, Content: "What is the capital of France?"},
		{Role: RoleAssistant, Content: "The capital of France is Paris."},
	}

	proc, err := o.Spawn(agent, WithMessages(history))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	// Verify messages were initialized
	proc.mu.RLock()
	if len(proc.messages) != 2 {
		t.Errorf("Process should have 2 initial messages, got %d", len(proc.messages))
	}
	if proc.messages[0].Content != "What is the capital of France?" {
		t.Errorf("First message content mismatch")
	}
	if proc.messages[1].Role != RoleAssistant {
		t.Errorf("Second message role mismatch")
	}
	proc.mu.RUnlock()
}

func TestWithMessagesEmpty(t *testing.T) {
	llm := &mockLLM{response: "Hello!"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	proc, err := o.Spawn(agent, WithMessages([]Message{}))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	proc.mu.RLock()
	if len(proc.messages) != 0 {
		t.Errorf("Process should have 0 messages, got %d", len(proc.messages))
	}
	proc.mu.RUnlock()
}

func TestWithMessagesNil(t *testing.T) {
	llm := &mockLLM{response: "Hello!"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	proc, err := o.Spawn(agent, WithMessages(nil))
	if err != nil {
		t.Fatalf("Spawn() returned error: %v", err)
	}

	// Should not panic and should have empty messages
	proc.mu.RLock()
	if proc.messages == nil {
		t.Error("Process messages should be initialized, not nil")
	}
	proc.mu.RUnlock()
}

func TestWithMessagesCopiesSlice(t *testing.T) {
	llm := &mockLLM{response: "Hello!"}
	o := NewOrchestrator(WithLLM(llm))

	agent := Agent{Name: "test"}
	original := []Message{
		{Role: RoleUser, Content: "Hello"},
	}

	proc, _ := o.Spawn(agent, WithMessages(original))

	// Modify original slice
	original[0].Content = "Modified"

	// Process should have the original value
	proc.mu.RLock()
	if proc.messages[0].Content != "Hello" {
		t.Error("WithMessages should copy the slice, not reference it")
	}
	proc.mu.RUnlock()
}
